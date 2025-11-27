#!/usr/bin/env bash
apt update
pip install uv
uv sync
source .venv/bin/activate
mkdir -p .venv/lib/python3.12/site-packages/assets
curl -L -o .venv/lib/python3.12/site-packages/assets/bpe_simple_vocab_16e6.txt.gz \
    "https://raw.githubusercontent.com/facebookresearch/sam3/refs/heads/main/assets/bpe_simple_vocab_16e6.txt.gz"
touch temp.ipynb
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension charliermarsh.ruff
[ -f ~/.cache/huggingface/token ] || huggingface-cli login
