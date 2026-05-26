"""Smoke-test the Gemini relabel prompt + schema on a handful of representative
training samples. Picks 5 diverse cases:

  1. Common: changeflag=1 with `building` + `road`.
  2. Rare class: changeflag=1 with `plant` or `asphalt` (long-tail).
  3. No-change: changeflag=0 (sample with all-none labels).
  4. Reversed-pair: a `_ters_` filename (A/B swap).
  5. Ambiguous attribute: contains `middle` (the semantically unclear one).

Prints (a) raw Gemini response, (b) parsed structured output, (c) confidence,
(d) whether schema validation passed, (e) whether any labels were dropped
by the sanitizer.

Run from repo root:
    python -m src.scripts.smoke_test_gemini
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    # Lazy imports for friendly error if deps missing
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
        sys.exit("GEMINI_API_KEY not in env. Add to .env or export it.")

    from src.scripts.gemini_relabel import (
        OBJECT_VOCAB, EVENT_VOCAB, ATTRIBUTE_VOCAB,
        build_prompt, sanitize_output,
    )

    # Same schema as gemini_relabel.py
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

    # Load dataset and pick 5 diverse samples
    ds = json.load(open("dataset/dataset.json"))["images"]
    train = [s for s in ds if s["split"] == "train"]

    chosen = []
    # 1. Common: changeflag=1 with building + road
    for s in train:
        if s["changeflag"] == 1 and "building" in s["object_labels"] and "road" in s["object_labels"]:
            chosen.append(("common", s)); break
    # 2. Rare class
    for s in train:
        if "plant" in s.get("object_labels", []) or "asphalt" in s.get("object_labels", []):
            chosen.append(("rare_class", s)); break
    # 3. No-change
    for s in train:
        if s["changeflag"] == 0:
            chosen.append(("no_change", s)); break
    # 4. _ters_ reversed pair
    for s in train:
        if "_ters_" in s["filename"]:
            chosen.append(("ters_reversed", s)); break
    # 5. Ambiguous middle
    for s in train:
        if "middle" in s.get("attribute_labels", []):
            chosen.append(("middle_ambiguous", s)); break

    print(f"Picked {len(chosen)} diverse samples:")
    for tag, s in chosen:
        print(f"  [{tag:18s}] {s['sample_id']}  obj={s.get('object_labels')}  evt={s.get('event_labels')}  attr={s.get('attribute_labels')}  cf={s.get('changeflag')}")

    client = genai.Client()
    t_total = time.time()
    schema_valid_count = 0
    dropped_count = 0
    results = []

    for i, (tag, s) in enumerate(chosen):
        a_path = Path("dataset") / s["rgb_A"]
        b_path = Path("dataset") / s["rgb_B"]
        if not a_path.exists() or not b_path.exists():
            print(f"\n[{tag}] SKIP — image not found locally ({a_path})")
            continue
        with open(a_path, "rb") as f: a_bytes = f.read()
        with open(b_path, "rb") as f: b_bytes = f.read()

        prompt = build_prompt(
            s.get("object_labels", []),
            s.get("event_labels", []),
            s.get("attribute_labels", []),
        )

        print(f"\n{'='*100}")
        print(f"[{tag}] {s['sample_id']}   changeflag={s.get('changeflag')}")
        print(f"  current labels: obj={s.get('object_labels')}  evt={s.get('event_labels')}  attr={s.get('attribute_labels')}")
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model="gemini-3.5-flash",
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
        except Exception as e:
            print(f"  ✗ API error: {e}")
            results.append({"tag": tag, "sample_id": s["sample_id"], "error": str(e)})
            continue
        elapsed = time.time() - t0

        # Schema-validated parse?
        if resp.parsed is not None:
            gem = resp.parsed.model_dump()
            schema_valid = True
            schema_valid_count += 1
        else:
            try:
                raw = json.loads(resp.text or "{}")
            except json.JSONDecodeError:
                raw = {}
            gem = sanitize_output(raw)
            schema_valid = False

        # Even with schema-valid parse, run sanitize to catch any "none" sneakery
        sanitized = sanitize_output(gem)
        n_dropped = sum(len(v) for v in sanitized.get("_dropped_invalid_labels", {}).values())
        if n_dropped:
            dropped_count += 1

        print(f"  API ms: {elapsed*1000:.0f}   schema_valid: {schema_valid}   dropped_invalid_labels: {n_dropped}")
        print(f"  Gemini parsed:")
        print(f"    object_labels:    {gem.get('object_labels')}")
        print(f"    event_labels:     {gem.get('event_labels')}")
        print(f"    attribute_labels: {gem.get('attribute_labels')}")
        print(f"    confidence:       {gem.get('confidence')}")
        print(f"    reasoning:        {gem.get('reasoning')}")
        if not schema_valid:
            print(f"  RAW text (first 500 chars): {(resp.text or '')[:500]}")
        results.append({
            "tag": tag,
            "sample_id": s["sample_id"],
            "current": {k: s.get(f"{k}_labels", []) for k in ("object", "event", "attribute")},
            "gemini": gem,
            "schema_valid": schema_valid,
            "dropped": sanitized.get("_dropped_invalid_labels", {}),
            "elapsed_ms": int(elapsed * 1000),
        })

    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"  samples processed:        {len(results)}")
    print(f"  schema-valid responses:   {schema_valid_count}/{len(results)}")
    print(f"  responses with dropped labels: {dropped_count}/{len(results)}")
    print(f"  total wall time:          {time.time() - t_total:.1f}s")
    out_dir = Path("results/_smoke_gemini")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "responses.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  full results → {out}")

    # ALSO copy the input image pairs side-by-side as a single PNG per sample
    # so the user can visually verify Gemini's claims.
    from PIL import Image
    for r, (tag, s) in zip(results, chosen):
        if "error" in r:
            continue
        a = Image.open(Path("dataset") / s["rgb_A"]).convert("RGB")
        b = Image.open(Path("dataset") / s["rgb_B"]).convert("RGB")
        canvas = Image.new("RGB", (a.width + b.width + 10, max(a.height, b.height)), (255, 255, 255))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + 10, 0))
        canvas.save(out_dir / f"{tag}__{s['sample_id']}.png")
    print(f"  Side-by-side BEFORE|AFTER PNGs → {out_dir}/  (open them to verify Gemini's claims)")


if __name__ == "__main__":
    main()
