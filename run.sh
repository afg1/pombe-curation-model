#! usr/bin/env bash


bash fetch_data.sh

python process_data.py

python five_fold.py 0 232011
python five_fold.py 1 232011
python five_fold.py 2 232011
python five_fold.py 3 232011
python five_fold.py 4 232011