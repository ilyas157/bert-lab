#!/bin/bash

module load cesga/2022 python/3.10.8
python -m  venv  $STORE/venvs/bertlab
source $STORE/venvs/bertlab/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
