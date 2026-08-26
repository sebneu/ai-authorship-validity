#!/bin/bash
cd ~/ai-authorship
export HF_HUB_OFFLINE=1
R() { echo "=== $(date +%H:%M) $* ==="; .venv/bin/python analysis/02_detectors/run_detectors.py \
        --corpus data/processed/corpus_v1/corpus.parquet "$@" 2>&1 | grep -viE "loading|checkpoint"; }
R --detectors fast_detect_gpt --batch-size 16
R --detectors binoculars      --batch-size 8
R --detectors detect_code_gpt --batch-size 16 --genres diff
echo "=== $(date +%H:%M) SWEEP FERTIG ==="
