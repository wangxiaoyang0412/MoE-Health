#!/usr/bin/env bash
# One-time environment setup (run from the project root): Python 3.11 venv, torch 2.7.1 (CUDA 12.6).
set -euo pipefail

UV="${UV:-$(command -v uv || echo "${HOME}/.local/bin/uv")}"
if [ ! -x "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    UV="${HOME}/.local/bin/uv"
fi

"$UV" venv --python 3.11 .venv
"$UV" pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
"$UV" pip install --python .venv/bin/python -e ".[dev]"
mkdir -p outputs
"$UV" pip freeze --python .venv/bin/python > outputs/requirements.lock.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
