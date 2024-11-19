#! usr/bin/env bash

python run_evaluation.py afg1/pombe-canto-data afg1/pombe_curation_fold_0 abstract label cuda
python run_evaluation.py afg1/pombe-canto-data afg1/pombe_curation_fold_1 abstract label cuda
python run_evaluation.py afg1/pombe-canto-data afg1/pombe_curation_fold_2 abstract label cuda
python run_evaluation.py afg1/pombe-canto-data afg1/pombe_curation_fold_3 abstract label cuda
python run_evaluation.py afg1/pombe-canto-data afg1/pombe_curation_fold_4 abstract label cuda --stop