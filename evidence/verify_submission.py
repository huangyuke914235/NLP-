"""Check the predictions and metrics accompanying the course reports."""
from pathlib import Path
from verify_predictions import score_set

ROOT=Path(__file__).resolve().parents[1]/'results'
for name in ('review_run_1','pair_run_1','truncation_run_1'):
    result=score_set(ROOT/name)
    print(f'{name}: {len(result)} sets of metrics recalculated from predictions')
