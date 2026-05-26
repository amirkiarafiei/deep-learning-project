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

# Note on image ordering: we send the two images as separate Part objects
# in the request, interleaved with explicit text headers "BEFORE IMAGE:" /
# "AFTER IMAGE:" so Gemini can never misread the order. This replaces the
# old ambiguous "first / second" wording in the prompt.

PROMPT_TEMPLATE = """You are an expert remote-sensing analyst. You have just seen two co-registered aerial RGB images of the same geographic location, shown above as `BEFORE IMAGE (time t1):` and `AFTER IMAGE (time t2):`. Both are 256x256 views of the same scene from above.

Your job: identify what visibly CHANGED between BEFORE and AFTER.

CRITICAL CALIBRATION RULES — read carefully, this is where most analysts make mistakes:

1. **Approximately 28% of pairs in this dataset have NO MEANINGFUL CHANGE.** They look essentially identical apart from minor lighting, registration, or sensor noise. When you cannot point to a specific, persistent, large-enough-to-name structural change between the two images, you MUST return `["none"]` for ALL three families with confidence="unknown".

2. **Do NOT invent or speculate.** If you have to use phrases like "appears to", "might be", "could be", "looks like" in your reasoning, that is a signal to set confidence="unknown" and return ["none"]. Reserve "high" for unambiguous, pixel-level-visible changes you could circle on the image.

3. **Do NOT pattern-match from typical remote-sensing change types.** Just because two images of a city often show construction does not mean THIS pair does. Look at the actual pixels.

4. **The current human labels may be wrong but they are NOT systematically zero.** If the current label says `["none"]` and the changeflag is 0, the prior is strongly that nothing changed — do not flip without unambiguous visible evidence.

OUTPUT FORMAT (strict)

Return exactly one JSON object with these five fields and no extra text:

  object_labels:    list of strings drawn ONLY from the OBJECT vocab below
  event_labels:     list of strings drawn ONLY from the EVENT vocab below
  attribute_labels: list of strings drawn ONLY from the ATTRIBUTE vocab below
  confidence:       one of "high" | "medium" | "low" | "unknown"
  reasoning:        ONE short English sentence, max 25 words, factual (no "appears to / might be / could be")

OBJECT vocab (closed): {object_vocab}, none
EVENT vocab (closed): {event_vocab}, none
ATTRIBUTE vocab (closed): {attribute_vocab}, none

CONFIDENCE LADDER
  "high"     = I can clearly point to the changed region; the change covers a non-trivial portion of the image; my description would survive cross-examination.
  "medium"   = I see a probable change but the boundary or class is ambiguous (e.g., "is that a building or a parking lot?").
  "low"      = I see a faint or partial difference, possibly real, possibly noise.
  "unknown"  = I cannot determine whether anything meaningful changed.

DEFAULTS WHEN UNSURE
  - confidence = "unknown"  →  return ["none"] in ALL three families.
  - confidence = "low" with truly minor changes  →  list at most ONE label per family.
  - Multi-label only when MULTIPLE genuine changes are visible.

CURRENT HUMAN LABELS (may be incomplete or wrong):
  object:    {curr_obj}
  event:     {curr_evt}
  attribute: {curr_attr}

A separately-trained model flagged these possible disagreements with the current labels:
  {disagreements}

EXAMPLES (illustrative — DO NOT copy labels; these are not your image):

Example 1 — clear single change (high confidence):
  {{"object_labels": ["building"], "event_labels": ["build"], "attribute_labels": ["large", "gray"],
    "confidence": "high",
    "reasoning": "A large gray building was constructed in the upper-left, on previously empty land."}}

Example 2 — no meaningful change (the common case for ~28% of pairs):
  {{"object_labels": ["none"], "event_labels": ["none"], "attribute_labels": ["none"],
    "confidence": "unknown",
    "reasoning": "Both images show the same residential block with no structural difference."}}

Example 3 — ambiguous (medium confidence, single label):
  {{"object_labels": ["road"], "event_labels": ["turn"], "attribute_labels": ["gray"],
    "confidence": "medium",
    "reasoning": "A small road segment in the center appears resurfaced from brown to gray."}}
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


def sanitize_output(parsed_dict: dict) -> dict:
    """Validate and clamp the Gemini response to the closed vocabulary.

    Always returns a dict with all 5 expected keys. Invalid labels are
    silently dropped (and counted). If a family ends up empty after
    dropping invalid labels, defaults to ["none"].
    """
    out: dict = {}
    dropped: dict = {"object": [], "event": [], "attribute": []}
    for fam, vocab in (
        ("object", OBJECT_VOCAB), ("event", EVENT_VOCAB), ("attribute", ATTRIBUTE_VOCAB)
    ):
        key = f"{fam}_labels"
        raw = parsed_dict.get(key, [])
        if not isinstance(raw, list):
            raw = [raw] if isinstance(raw, str) else []
        # Lower-case + strip + filter against the vocab (+ "none")
        clean = []
        vocab_set = set(vocab) | {"none"}
        for lbl in raw:
            if not isinstance(lbl, str):
                continue
            lbl_norm = lbl.lower().strip()
            if lbl_norm in vocab_set:
                if lbl_norm not in clean:  # dedup
                    clean.append(lbl_norm)
            else:
                dropped[fam].append(lbl)
        if not clean:
            clean = ["none"]
        out[key] = clean

    conf = parsed_dict.get("confidence", "unknown")
    if isinstance(conf, str):
        conf = conf.lower().strip()
    if conf not in {"high", "medium", "low", "unknown"}:
        conf = "unknown"
    out["confidence"] = conf

    reasoning = parsed_dict.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    out["reasoning"] = reasoning.strip()[:300]  # hard cap

    out["_dropped_invalid_labels"] = {k: v for k, v in dropped.items() if v}
    return out


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
        from typing import Literal
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Run: pip install google-genai python-dotenv pydantic")

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")

    # Pydantic schema with Literal-typed vocabularies. google-genai converts
    # Literal["a", "b", ...] into JSON Schema enums, which constrains
    # Gemini's structured output to the exact closed vocabulary.
    ObjectLabel = Literal[
        "building", "tree", "road", "field", "vegetation", "water",
        "parking", "land", "roof", "asphalt", "green", "plant", "none",
    ]
    EventLabel = Literal[
        "build", "remove", "turn", "appear", "replace", "change",
        "destroy", "increase", "vegetate", "add", "surround", "remain", "none",
    ]
    AttributeLabel = Literal[
        "blue", "gray", "green", "large", "huge", "black", "white",
        "more", "small", "brown", "empty", "bare", "lush", "middle",
        "red", "residential", "long", "industrial", "adjacent",
        "sparse", "dense", "paved", "same", "dark", "none",
    ]
    Confidence = Literal["high", "medium", "low", "unknown"]

    class LabelOutput(BaseModel):
        object_labels: list[ObjectLabel]
        event_labels: list[EventLabel]
        attribute_labels: list[AttributeLabel]
        confidence: Confidence
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
                # Interleave text labels with images so the order is unambiguous.
                resp = client.models.generate_content(
                    model=args.model,
                    contents=[
                        "BEFORE IMAGE (time t1):",
                        types.Part.from_bytes(data=a_bytes, mime_type="image/png"),
                        "AFTER IMAGE (time t2):",
                        types.Part.from_bytes(data=b_bytes, mime_type="image/png"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LabelOutput,
                    ),
                )
                parsed = resp.parsed
                # Build the gemini dict from the parsed Pydantic instance (schema-valid)
                # OR from a JSON parse + sanitize of the raw text (schema-invalid fallback).
                if parsed is not None:
                    gemini_dict = parsed.model_dump()
                    gemini_dict["_schema_valid"] = True
                else:
                    try:
                        raw_dict = json.loads(resp.text or "{}")
                    except json.JSONDecodeError:
                        raw_dict = {}
                    gemini_dict = sanitize_output(raw_dict)
                    gemini_dict["_schema_valid"] = False
                    gemini_dict["_raw"] = (resp.text or "")[:1000]

                record = {
                    "sample_id": sid,
                    "current": {
                        "object_labels": s.get("object_labels", []),
                        "event_labels": s.get("event_labels", []),
                        "attribute_labels": s.get("attribute_labels", []),
                    },
                    "gemini": gemini_dict,
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
