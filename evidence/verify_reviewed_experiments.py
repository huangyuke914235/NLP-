"""Verify active transfer mapping and recalculate fixed-answer diagnostics."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def js(path):
    return json.loads(path.read_text('utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual(path, official=None, curated=None):
    returned=defaultdict(list)
    for row in rows(path):
        if official is not None:
            entry=official[row['id']]
            assert row['original_question']==entry['question'] and row['fixed_answer']==entry['answer']
            assert row['gold']==entry['label']
        if curated is not None:
            case=curated[row['id']]
            assert all(row[k]==v for k,v in case.items()),row['id']
        returned[row['model']].append(row)
    first=next(iter(returned.values()))
    template={r['id']:(r['gold'],r['changed_expected'],r['change_kind']) for r in first}
    support=Counter(r['changed_expected'] for r in first)
    kinds=Counter(r['change_kind'] for r in first)
    report={}
    for model, rr in returned.items():
        assert len(rr)==len(template) and {r['id']:(r['gold'],r['changed_expected'],r['change_kind']) for r in rr}==template
        baseline=[r for r in rr if r['gold']==r['original_prediction']]
        required=[r for r in rr if r['gold']!=r['changed_expected']]
        assert all((r['gold_requires_change']=='True')==(r in required) for r in rr)
        for r in rr:
            for var in ('original','paraphrase','changed'):
                prob={label:float(r[f'{var}_prob_{label}']) for label in support}
                assert abs(sum(prob.values())-1)<1e-6 and max(prob.values())==prob[r[f'{var}_prediction']]
        correct=lambda r:r['changed_prediction']==r['changed_expected']
        act=lambda r:correct(r) and r['changed_prediction']!=r['original_prediction']
        req_with_base=[r for r in required if r in baseline]
        report[model]={
            'n':len(rr),'original_correct_n':len(baseline),
            'paraphrase_stable_n':sum(r['original_prediction']==r['paraphrase_prediction'] for r in rr),
            'paraphrase_both_correct_n':sum(r['gold']==r['paraphrase_prediction'] for r in baseline),
            'changed_expected_label_correct_n':sum(map(correct,rr)),
            'gold_requires_change_n':len(required),'correct_and_changed_on_required_n':sum(map(act,required)),
            'required_and_original_correct_n':len(req_with_base),
            'correct_change_given_original_correct_n':sum(map(act,req_with_base)),
            'same_scoped_refusal_preserved_n':sum(correct(r) for r in rr if r['change_kind']=='same_scoped_refusal'),
            'changed_correct_by_expected':{label:sum(correct(r) for r in rr if r['changed_expected']==label) for label in support},
        }
    return {'changed_expected_support':dict(support),'change_kind_support':dict(kinds),'models':report}


def prediction_path(directory):
    for name in ('counterfactual.csv','counterfactual_public.csv'):
        p=directory/name
        if p.is_file(): return p
    raise FileNotFoundError(directory/'counterfactual.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,default=Path('results/run_1'))
    p.add_argument('--first',type=Path,default=Path('results/review_run_1'))
    p.add_argument('--second',type=Path,default=Path('results/review_run_2'))
    p.add_argument('--out',type=Path,default=Path('evidence/reviewed_experiment_audit.json'))
    args=p.parse_args()
    original=js(args.before/'metrics.json'); current=js(args.first/'metrics.json')
    replay=js(args.second/'metrics.json')
    assert current==replay and set(current)-set(original)=={'target_direct_inclusive'}
    equal={}
    for key,m in original.items():
        assert m==current[key], key
        a=args.before/m['prediction_file']; b=args.first/m['prediction_file']; c=args.second/m['prediction_file']
        assert sha(a)==sha(b)==sha(c),key
        equal[key]=sha(a)
    expanded='target_direct_inclusive'
    rel=current[expanded]['prediction_file']
    assert sha(args.first/rel)==sha(args.second/rel)
    assert js(args.first/'adapter_used.json')==js(args.second/'adapter_used.json')==js(Path('configs/adapter.json'))
    assert current['target_direct']['parameters']['label_mapping']==js(args.first/'adapter_used.json')['direct_transfer']
    assert 'transfer_label_mapping' not in js(Path('configs/experiment.json'))
    first=prediction_path(args.first); replay_cf=prediction_path(args.second)
    assert sha(first)==sha(replay_cf)
    have_text=first.name=='counterfactual.csv'
    official=curated=None
    if have_text:
        official={r['id']:r for r in rows(Path('data/processed/qevasion_grouped.csv')) if r['split']=='test'}
        curated={r['id']:r for r in rows(Path('configs/counterfactual_reviewed.csv'))}
    ref=actual(first,official,curated)
    assert ref==js(args.first/'counterfactual.json')==js(args.second/'counterfactual.json')
    assert ref['changed_expected_support']=={'Ambivalent':12,'Clear Reply':4,'Clear Non-Reply':3}
    audit={'reference_prediction_files_byte_equal':len(equal), 'baseline_sha256':equal,
           'new_prediction_files_equal':{expanded:sha(args.first/rel)}, 'adapter_sha256':sha(Path('configs/adapter.json')),
           'counterfactual_recomputed':ref,'counterfactual_prediction_sha256':sha(first), 'full_fixed_answer_checked':have_text}
    args.out.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(f'{len(equal)} unchanged reference predictions and one new mapping sensitivity reproduced; verified {sum(ref["changed_expected_support"].values())} counterfactual cases.')


if __name__=='__main__': main()
