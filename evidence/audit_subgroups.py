"""Audit error cases, model behavior under fixed-answer edits, and subgroup counts."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def dump(path, payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sums(cases):
    return {'n':len(cases),'correct':sum(r['gold']==r['prediction'] for r in cases)}


def main():
    run = ROOT/'results/run_1'
    out = ROOT/'evidence'
    data = {r['id']:r for r in rows(ROOT/'data/processed/qevasion_grouped.csv') if r['split']=='test'}
    pred = {r['id']:r for r in rows(run/'predictions/grouped_tfidf.csv')}
    error_cases = []
    for note in rows(out/'error_notes.csv'):
        ident = note['id']; scored = pred[ident]; actual = data[ident]
        assert scored['gold']==actual['label'] and scored['gold']!=scored['prediction'], ident
        error_cases.append(dict(note, gold=scored['gold'],prediction=scored['prediction'],
                                confidence=float(scored['prob_'+scored['prediction']]),group=actual['group'],
                                answer_length=int(scored['answer_length'])))
    assert len(error_cases)>=10 and len({x['id'] for x in error_cases})==len(error_cases)
    dump(out/'verified_error_cases.json',error_cases)
    subgroups = {}
    for key in ('grouped_tfidf','grouped_minilm','grouped_minilm_interactions','official_tfidf'):
        raw = rows(run/f'predictions/{key}.csv')
        lengths = {'<100': [],'100–299': [],'≥300': []}
        interviews = defaultdict(list)
        train_groups = {r['group'] for r in rows(ROOT/'data/processed/qevasion_official.csv') if r['split']=='train'}
        test_seen = {'official_train_url_seen': [],'official_train_url_unseen': []}
        for r in raw:
            n=int(r['answer_length']); lengths['<100' if n<100 else '100–299' if n<300 else '≥300'].append(r)
            interviews[r['group']].append(r)
            test_seen['official_train_url_seen' if r['group'] in train_groups else 'official_train_url_unseen'].append(r)
        subgroups[key]={'length':{k:sums(v) for k,v in lengths.items()},
                        'interview_support_at_least_8':{k:sums(v) for k,v in sorted(interviews.items()) if len(v)>=8},
                        'common_test_seen_by_official_training':{k:sums(v) for k,v in test_seen.items()}}
    cf = defaultdict(list)
    fixed_questions = {r['id']:r for r in rows(ROOT/'configs/counterfactual.csv')}
    for r in rows(run/'counterfactual.csv'):
        orig=data[r['id']]; config=fixed_questions[r['id']]
        assert r['gold']==orig['label'] and r['fixed_answer']==orig['answer'] and r['original_question']==orig['question']
        assert r['paraphrase']==config['paraphrase'] and r['unrelated']==config['unrelated']
        assert r['unrelated_expected']==config['unrelated_expected']
        cf[r['model']].append(r)
    cf_checked = {}
    for key, group in cf.items():
        assert set(r['id'] for r in group)==fixed_questions.keys()
        original={r['id']:r['prediction'] for r in rows(run/f'predictions/{key}.csv')}
        assert all(r['original_prediction']==original[r['id']] for r in group)
        original_correct=[r for r in group if r['gold']==r['original_prediction']]
        should_change=[r for r in group if r['gold']!=r['unrelated_expected']]
        fixed_right=lambda r: r['unrelated_prediction']==r['unrelated_expected']
        changes=lambda r: r['original_prediction']!=r['unrelated_prediction']
        conditional=[r for r in should_change if r['original_prediction']==r['gold']]
        cf_checked[key]={
            'n':len(group),'original_correct':len(original_correct),'invariant':sum(r['original_prediction']==r['paraphrase_prediction'] for r in group),
            'both_gold_correct':sum(r['paraphrase_prediction']==r['gold'] for r in original_correct),
            'unrelated_correct':sum(fixed_right(r) for r in group),'should_change':len(should_change),
            'unrelated_both_correct_and_changes':sum(fixed_right(r) and changes(r) for r in should_change),
            'should_change_and_original_correct':len(conditional),
            'correct_changes_among_initially_correct':sum(fixed_right(r) and changes(r) for r in conditional),
        }
    dump(out/'subgroup_and_counterfactual_audit.json',{'subgroups':subgroups,'counterfactual':cf_checked})
    print('verified IDs of',len(error_cases),'real errors and',len(fixed_questions),'fixed-answer triples')


if __name__=='__main__': main()
