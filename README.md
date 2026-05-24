# deep-learning-project

Semester project for **BLM5135 — Neural Networks and Deep Learning**
(Yıldız Technical University, Spring 2026).

**Task:** scene-level multi-label classification of bi-temporal
remote-sensing image pairs. Given two co-registered RGB images of the
same scene at times *t₁* (before) and *t₂* (after), predict three
multi-label outputs — Object, Event, Attribute — describing what
changed. Data is derived from LEVIR-CC (Liu et al., TGRS 2022).

## Where to look

| You want… | Read |
|---|---|
| Project context, scope, non-goals | [`AGENTS.md`](./AGENTS.md) |
| Architecture & implementation spec | [`docs/track1.md`](./docs/track1.md) |
| Course rubric (Turkish PDF) | [`docs/clasroom/`](./docs/clasroom/) |
| Dataset statistics | [`results/eda/eda_summary.md`](./results/eda/eda_summary.md) |
| Canonical label ordering | [`results/eda/label_vocab.json`](./results/eda/label_vocab.json) |
| Literature reviews | [`docs/literature/`](./docs/literature/) |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Reproduce the EDA — requires dataset/ locally.
python scripts/run_eda.py
```

Training runs on Google Colab Pro (A100), not locally — see
[`AGENTS.md`](./AGENTS.md) § Where compute runs.

## License

[MIT](./LICENSE).
