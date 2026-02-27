# Explainable AI for Alarm Decision Support in Control Room Settings
Transformer‑based alarm decision support with Explainable AI for control room settings


> Transformer-based anomaly detection for control-room alarms with Explainable AI. Includes technical evaluation (fidelity, robustness, latency) and a human-centred user study (SART & SPAM).

<p align="center">
  <a href="https://github.com/arunbaruah/Anomaly_Detection_Transformer">Base Transformer implementation</a> ·
  <a href="./docs/research-overview.md">Research overview</a> ·
  <a href="./docs/getting-started.md">Getting started</a>
</p>

## Why this project?

Control rooms in high-stakes domains face alarm floods and data overload. We implement a **Transformer** anomaly model with **explanations** and evaluate both technically and with a user study (trust, workload, situational awareness). (Research background derived from our lit review & proposal.)¹

## Model: Transformer

**Transformer-based anomaly detector**, adapted from `arunbaruah/Anomaly_Detection_Transformer`.² We wrap it to operate on alarm/event sequences and provide XAI views.

- Upstream repo (logs/Transformer, PyTorch).²
- We expose a simple interface: `cr_xai.models.transformer_wrapper.TransformerAlarmModel`.

## Features

- **Transformer anomaly modeling** (unsupervised next-token/log template prediction thresholding)
- **Explainability**:
- **Evaluations**:
  - Technical: fidelity, stability, robustness, **latency** (feasibility during floods)
  - Human-centred: SART + SPAM probes, decision time/accuracy, calibrated trust
- **Docs site** (MkDocs + Material) and CI
- **Reproducible env**: `environment.yml`

## Getting started

```bash
conda env create -f environment.yml
conda activate cr-xai
# fetch example tiny data (synthetic)
python -m cr_xai.datasets.download_example
# train (toy)
python src/train.py --config src/cr_xai/configs/baseline.yaml
# evaluate + generate explanations
python src/evaluate.py --config src/cr_xai/configs/baseline.yaml --explain
```

## Folder Layout

```text
controlroom-xai-alarm-support-transformer/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
├─ .gitignore
├─ environment.yml           # or requirements.txt
├─ pyproject.toml            # optional if you package src/
├─ data/
│  ├─ README.md              # where to download datasets; no raw data committed
│  ├─ example/               # tiny synthetic snippets for tests only
├─ src/
│  ├─ cr_xai/                # your package
│  │  ├─ __init__.py
│  │  ├─ configs/            # yaml configs (paths, hyperparams)
│  │  ├─ datasets/           # dataset loaders, parsers
│  │  ├─ models/             # wrappers around the Transformer
│  │  ├─ explainability/     # SHAP, counterfactuals (CAFE-like)
│  │  ├─ evaluation/         # metrics, robustness, latency
│  │  ├─ ui/                 # simple Dash/Gradio prototype for presentations
│  │  └─ utils/              # logging, seeds, timing
│  ├─ train.py               # training entry point
│  ├─ evaluate.py            # technical evaluation
│  └─ userstudy/             # SPAM/SART probes, logging, consent stubs
├─ notebooks/
│  ├─ 00_exploration.ipynb
│  ├─ 10_training_demo.ipynb
│  └─ 20_xai_examples.ipynb
├─ third_party/
│  └─ Anomaly_Detection_Transformer/   # as git submodule or fork
├─ tests/
│  ├─ test_data_loading.py
│  ├─ test_model_forward.py
│  └─ test_explainers.py
├─ docs/
│  ├─ index.md
│  ├─ getting-started.md
│  ├─ research-overview.md
│  ├─ user-study.md
│  ├─ xai-methods.md
│  └─ api.md
├─ mkdocs.yml
└─ .github/
   ├─ workflows/
   │  ├─ ci.yml              # lint + tests
   │  └─ docs.yml            # publish docs site
   └─ ISSUE_TEMPLATE/
      ├─ bug_report.md
      └─ feature_request.md
```
