#! usr/bin/env bash


bash fetch_data.sh

python process_data.py

python five_fold.py 0
python five_fold.py 1
python five_fold.py 2
python five_fold.py 3
python five_fold.py 4