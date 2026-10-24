#!/usr/bin/env bash
# One-step environment setup. Run from the repo root: ./setup.sh
set -euo pipefail

XL_LEXEME="git+https://github.com/pierluigic/xl-lexeme.git@0f6748df5c3c444dde8e98313b15775b1b22c50c"

cd "$(dirname "$0")"

# Find a working Python 3.11 (plain install, Homebrew, or pyenv)
PYTHON=""
for candidate in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 \
                 "$HOME"/.pyenv/versions/3.11*/bin/python3.11; do
  if "$candidate" -c 'import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.11 not found. Install it first:"
  echo "  macOS:  brew install python@3.11"
  echo "  Linux:  sudo apt install python3.11 python3.11-venv"
  exit 1
fi
echo "Using $PYTHON"

if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# xl-lexeme's setup.py imports pip, so it must build inside this environment
pip install --no-build-isolation --no-deps "$XL_LEXEME"

python -c "import torch, WordTransformer; print('torch', torch.__version__, '| MPS:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"

echo
echo "Done. Activate the environment with:  source .venv/bin/activate"
