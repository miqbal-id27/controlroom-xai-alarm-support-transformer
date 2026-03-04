# 1) Create repo & initialize
git init controlroom-xai-alarm-support-transformer
cd controlroom-xai-alarm-support-transformer

# 2) Add upstream Transformer as a submodule
git submodule add https://github.com/arunbaruah/Anomaly_Detection_Transformer third_party/Anomaly_Detection_Transformer

# 3) Add base files (README, LICENSE, etc.)  -> paste from sections below
# 4) Create and activate environment
conda env create -f environment.yml
conda activate cr-xai

# 5) Run unit tests
pytest -q

# 6) Train/evaluate
python src/train.py --config src/cr_xai/configs/baseline.yaml
python src/evaluate.py --config src/cr_xai/configs/baseline.yaml

# 7) Build docs locally, then publish
mkdocs serve
# and when ready (GH Pages via Action will do it automatically)