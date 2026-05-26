"""Re-annotate ambiguous training samples and the validation set via Gemini.

Sends each candidate sample (image_A + image_B + current labels) to the
``gemini-3.5-flash`` model with a strict JSON schema asking for the model's
best annotation + confidence + reasoning. Outputs an append-only JSONL file
so a mid-run API interruption preserves progress.

Cost (Gemini 2.5 Flash standard): $0.30 / 1M input tokens, $2.50 / 1M output.
Per call ≈ 1500 input + 300 output tokens. 1500 calls ≈ $2 total.

Usage:
    # Top-K train candidates:
    python -m src.scripts.gemini_relabel \\
        --candidates results/track1_v3/cleanup/noisy_candidates.json \\
        --dataset-json dataset/dataset.json \\
        --dataset-root dataset \\
        --output results/track1_v3/cleanup/gemini_relabels_train.jsonl \\
        --max-samples 500

    # Entire val set:
    python -m src.scripts.gemini_relabel \\
        --split val \\
        --dataset-json dataset/dataset.json \\
        --dataset-root dataset \\
        --output results/track1_v3/cleanup/gemini_relabels_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Lazy import inside main() so the import error message is friendly when
# users skip `pip install -r requirements.txt`.


OBJECT_VOCAB = ["building", "tree", "road", "field", "vegetation", "water",
                "parking", "land", "roof", "asphalt", "green", "plant"]
EVENT_VOCAB = ["build", "remove", "turn", "appear", "replace", "change",
               "destroy", "increase", "vegetate", "add", "surround", "remain"]
ATTRIBUTE_VOCAB = ["blue", "gray", "green", "large", "huge", "black", "white",
                   "more", "small", "brown", "empty", "bare", "lush", "middle",
                   "red", "residential", "long", "industrial", "adjacent",
                   "sparse", "dense", "paved", "same", "dark"]

PROMPT_TEMPLATE = """You are a remote-sensing-imagery expert. You will see TWO co-registered aerial RGB images of the same location at TWO different times (the "before" image first, the "after" image second). Your task is to identify what visibly CHANGED between them.

Three independent label families describe the change. Each is a multi-label set drawn from a closed vocabulary; "none" means no meaningful change in that family.

OBJECT — what kind of thing changed (closed vocabulary):
  {object_vocab}

EVENT — the action / transition (closed vocabulary):
  {event_vocab}

ATTRIBUTE — descriptive property of the change (closed vocabulary):
  {attribute_vocab}

The current human-supplied labels are below. They may be incomplete, ambiguous, or wrong. The dataset is known to have label-quality issues.

  current object: {curr_obj}
  current event:  {curr_evt}
  current attribute: {curr_attr}

Possibly-disagreeing classes the predicting model flagged:
  {disagreements}

Output JSON with these exact fields:
  - object_labels: list of strings from the OBJECT vocab (empty list or ["none"] if no object change)
  - event_labels: list of strings from the EVENT vocab
  - attribute_labels: list of strings from the ATTRIBUTE vocab
  - confidence: one of "high", "medium", "low", "unknown"
  - reasoning: ONE short sentence

Rules:
- If the two images look identical or you cannot tell, use ["none"] in all three families, confidence="unknown".
- If you confidently disagree with the current labels (clear visible change they missed, or labeled change you don't see), set confidence="high" and provide YOUR labels.
- If you're not sure, set confidence="medium" or "low" and provide your best guess.
- Only use vocabulary terms listed above. Never invent new labels.
"""


def build_prompt(current_obj, current_evt, current_attr, disagreements=None) -> str:
    disagreements_str = "(none)" if not disagreements else "; ".join(
        f"{fam}: {', '.join(d)}" for fam, d in disagreements.items() if d
    ) or "(none)"
    return PROMPT_TEMPLATE.format(
        object_vocab=", ".join(OBJECT_VOCAB),
        event_vocab=", ".join(EVENT_VOCAB),
        attribute_vocab=", ".join(ATTRIBUTE_VOCAB),
        curr_obj=", ".join(current_obj) if current_obj else "none",
        curr_evt=", ".join(current_evt) if current_evt else "none",
        curr_attr=", ".join(current_attr) if current_attr else "none",
        disagreements=disagreements_str,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini re-annotation of uncertain samples")
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True, help="Append-only JSONL output file")
    parser.add_argument("--candidates", help="Path to noisy_candidates.json (train candidate mode)")
    parser.add_argument("--split", choices=["val", "test"], help="Re-label an entire split instead of candidates")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already present in output JSONL")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep seconds between calls (rate-limit safety)")
    args = parser.parse_args()

    # Lazy imports
    try:
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types
        from pydantic import BaseModel
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Run: pip install google-genai python-dotenv pydantic")

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")

    class LabelOutput(BaseModel):
        object_labels: list[str]
        event_labels: list[str]
        attribute_labels: list[str]
        confidence: str
        reasoning: str

    # Build the queue of sample ids to relabel
    dataset = json.load(open(args.dataset_json))["images"]
    by_id = {s["sample_id"]: s for s in dataset}
    queue: list[dict] = []

    if args.candidates:
        cand = json.load(open(args.candidates))
        for entry in cand["candidates"]:
            sid = entry["sample_id"]
            if sid in by_id:
                queue.append({
                    "sample": by_id[sid],
                    "disagreements": entry.get("disagreements", {}),
                })
    elif args.split:
        for s in dataset:
            if s["split"] == args.split:
                queue.append({"sample": s, "disagreements": None})
    else:
        sys.exit("Either --candidates or --split must be given.")

    if args.max_samples is not None:
        queue = queue[: args.max_samples]
    print(f"Queue size: {len(queue)}")

    # Resume support: skip already-done sample IDs
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} samples already in output.")

    client = genai.Client()  # reads GEMINI_API_KEY from env
    n_processed = 0
    n_errors = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for entry in queue:
            s = entry["sample"]
            sid = s["sample_id"]
            if sid in done:
                continue
            try:
                with open(Path(args.dataset_root) / s["rgb_A"], "rb") as f:
                    a_bytes = f.read()
                with open(Path(args.dataset_root) / s["rgb_B"], "rb") as f:
                    b_bytes = f.read()
                prompt = build_prompt(
                    s.get("object_labels", []),
                    s.get("event_labels", []),
                    s.get("attribute_labels", []),
                    entry.get("disagreements"),
                )
                resp = client.models.generate_content(
                    model=args.model,
                    contents=[
                        types.Part.from_bytes(data=a_bytes, mime_type="image/png"),
                        types.Part.from_bytes(data=b_bytes, mime_type="image/png"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LabelOutput,
                    ),
                )
                parsed = resp.parsed
                record = {
                    "sample_id": sid,
                    "current": {
                        "object_labels": s.get("object_labels", []),
                        "event_labels": s.get("event_labels", []),
                        "attribute_labels": s.get("attribute_labels", []),
                    },
                    "gemini": parsed.model_dump() if parsed else {"raw": resp.text},
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                n_processed += 1
                if n_processed % 25 == 0:
                    elapsed = time.time() - t0
                    rate = n_processed / max(elapsed, 1e-6)
                    print(f"  {n_processed}/{len(queue)-len(done)} done in {elapsed:.0f}s ({rate:.1f}/s)")
            except Exception as e:
                n_errors += 1
                print(f"  ERROR on {sid}: {e}")
                if n_errors > 10:
                    print("Too many errors — aborting.")
                    break
            time.sleep(args.sleep)
    print(f"\nDone. Processed {n_processed}, errors {n_errors}. Output: {out_path}")


if __name__ == "__main__":
    main()
