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

