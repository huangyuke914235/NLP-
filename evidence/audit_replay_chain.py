"""Confirm separately prepared datasets and supplemental evaluation outputs match."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def ids_and_labels(path):
    return {r['id']:r['gold'] for r in csv_rows(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-data',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('evidence/replay_chain.json'))
    args=p.parse_args()
    hashes={}
    for upstream in ('qevasion','qnli'):
        orig=read(Path('data/processed')/f'{upstream}_manifest.json')
        replay=read(args.replay_data/'processed'/f'{upstream}_manifest.json')
        assert orig==replay,upstream
        for name,value in orig['generated_files'].items():
            assert digest(Path('data/processed')/name)==digest(args.replay_data/'processed'/name)==value,name
            hashes[name]=value
        for record in orig['raw_files']:
            name=record['file']
            assert digest(Path('data/raw')/name)==digest(args.replay_data/'raw'/name)==record['sha256'],name
            hashes[name]=record['sha256']
    files=['counterfactual.csv','counterfactual.json','hierarchy.json','data_efficiency.json','data_efficiency_train_ids.csv','selection.json','metrics.json']
    base=Path('results/run_1'); second=Path('results/run_2')
    supplemental={name:digest(base/name)==digest(second/name) for name in files}
    supplemental['pair_selection.json']=digest(Path('results/pair_run_1/selection.json'))==digest(Path('results/pair_run_2/selection.json'))
    supplemental['pair_metrics.json']=digest(Path('results/pair_run_1/metrics.json'))==digest(Path('results/pair_run_2/metrics.json'))
    assert all(supplemental.values()),supplemental
    for group,ref in [('grouped','grouped_tfidf'),('target','target_direct')]:
        classifier=Path(f'results/pair_run_1/predictions/{group}_pair.csv')
        assert ids_and_labels(classifier)==ids_and_labels(base/f'predictions/{ref}.csv'),group
    efficiency=read(base/'data_efficiency.json')
    values=defaultdict(list)
    audit=read(Path('evidence/prediction_audit.json'))['recomputed']
    for row in efficiency['runs']:
        ref=audit[Path(row['prediction_file']).stem]
        assert abs(row['macro_f1']-ref['macro_f1'])<1e-9 and abs(row['accuracy']-ref['accuracy'])<1e-9
        values[row['fraction'],row['method']].append(ref['macro_f1'])
    for row in efficiency['summary']:
        observed=values[row['fraction'],row['method']]
        assert abs(row['mean_macro_f1']-statistics.mean(observed))<1e-9
        assert abs(row['sd_macro_f1']-statistics.stdev(observed))<1e-9
    datasets=defaultdict(set)
    for row in csv_rows(base/'data_efficiency_train_ids.csv'):
        datasets[int(row['seed']),float(row['fraction'])].add(row['id'])
    for seed in {k[0] for k in datasets}:
        assert datasets[seed,.1]<datasets[seed,.3]<datasets[seed,1.0]
    result={'raw_and_processed_sha256_identical':hashes,'byte_equal_supplemental_artifacts':supplemental,
            'pair_fixed_evaluation_ids_match_baselines':True,'efficiency_aggregates_recomputed':True,'efficiency_subsets_nested':True}
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Checks passed for {len(hashes)} raw/derived files and {len(supplemental)} supplemental outputs')


if __name__=='__main__': main()
