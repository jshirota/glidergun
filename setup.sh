#!/usr/bin/env bash
apt update
pip install uv
uv sync
source .venv/bin/activate
touch temp.ipynb
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension charliermarsh.ruff
[ -f ~/.cache/huggingface/token ] || huggingface-cli login
