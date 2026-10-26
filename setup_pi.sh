#!/usr/bin/env bash
# One-time setup on Raspberry Pi OS 64-bit (Bookworm or newer). Needs internet.
set -e
sudo apt update
sudo apt install -y python3-venv python3-dev
python3 -m venv "$HOME/quadpi-venv"
source "$HOME/quadpi-venv/bin/activate"
pip install --upgrade pip wheel
# Pinned versions (requirements.txt) if wheels exist for this Python; otherwise latest.
pip install numpy==2.5.3 scipy==1.18.1 osqp==1.1.3 || pip install numpy scipy osqp
python -c "import numpy,scipy,osqp;print('OK numpy',numpy.__version__,'scipy',scipy.__version__,'osqp',osqp.__version__)"
