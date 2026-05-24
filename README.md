# deep-learning-project

Semester project for **BLM5135 — Neural Networks and Deep Learning**.

Task: scene-level **multi-label classification of bi-temporal remote-sensing
image pairs**. Given two co-registered RGB images of the same scene at times
*t₁* and *t₂*, predict three multi-label outputs describing what changed:

| Family    | # classes | Example labels                |
|-----------|----------:|-------------------------------|
| Object    |        13 | building, road, tree, water   |
| Event     |        13 | build, remove, replace, turn  |
| Attribute |        25 | green, dark, large, residential |

See [`ReadMe.txt`](./ReadMe.txt) for end-user setup and run instructions
(the file the grader will read), and [`docs/`](./docs/) for the project
brief and literature review.

## Repository layout

```
.
├── ReadMe.txt              # grader-facing setup + run instructions
├── requirements.txt        # pinned Python dependencies (currently EDA-only)
├── scripts/
│   └── run_eda.py          # exploratory data analysis pipeline
├── docs/
│   ├── BLM5135-ProjeAçıklaması.pdf
│   ├── literature_codex.md
│   ├── literature_flash.md
│   └── literature_opus.md
├── results/
│   └── eda/                # produced by scripts/run_eda.py
└── dataset/                # local-only; provided by course (gitignored)
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_eda.py
```
