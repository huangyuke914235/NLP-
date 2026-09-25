"""Label hierarchy extension. Reads existing data/embeddings without changing them."""
import argparse, csv, hashlib, json, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
base = args.project.resolve()
out = args.output or base/'outputs'/'hierarchy'
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(base/'deps'))
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

COARSE = ['Clear Reply','Ambivalent','Clear Non-Reply']
FINE = ['Explicit','Implicit','Dodging','General','Deflection','Partial/half-answer','Declining to answer','Claims ignorance','Clarification']
MAP = {x: (0 if x == 'Explicit' else 2 if x in FINE[-3:] else 1) for x in FINE}
ds = load_from_disk(str(base/'data'/'qevasion_raw'))
tr, te = ds['train'].to_pandas(), ds['test'].to_pandas()
emb = np.load(base/'outputs'/'st_embeddings.npz')
X, T = emb['X_train'], emb['X_test']
y, yt = tr.clarity_label.map(dict(zip(COARSE,range(3)))).to_numpy(), te.clarity_label.map(dict(zip(COARSE,range(3)))).to_numpy()
assert np.array_equal(y,emb['y_train']) and np.array_equal(yt,emb['y_test'])
assert len(X)==len(tr) and len(T)==len(te)
fine = tr.evasion_label.map(dict(zip(FINE,range(9)))).to_numpy()
assert not pd.isna(fine).any()
def metrics(gold, pred, names):
    return dict(accuracy=float(accuracy_score(gold,pred)), macro_f1=float(f1_score(gold,pred,labels=range(len(names)),average='macro',zero_division=0)), per_class=classification_report(gold,pred,labels=list(range(len(names))),target_names=names,output_dict=True,zero_division=0), confusion_matrix=confusion_matrix(gold,pred,labels=list(range(len(names)))).tolist())
def fit(features, labels):
    return LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced',random_state=42).fit(features,labels)
def aggregate(model,features):
    p = model.predict_proba(features)
    coarse_p = np.zeros((len(features),3))
    for j,k in enumerate(model.classes_): coarse_p[:,MAP[FINE[int(k)]]] += p[:,j]
    return coarse_p.argmax(axis=1),coarse_p

fit_i,val_i = next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42).split(X,y,groups=tr.title))
direct = fit(X[fit_i], y[fit_i]); hierarchical = fit(X[fit_i], fine[fit_i])
val_direct = direct.predict(X[val_i]); val_fine = hierarchical.predict(X[val_i]); val_hier,_ = aggregate(hierarchical,X[val_i])
refit_direct = fit(X,y); refit_hier = fit(X,fine)
pred_d = refit_direct.predict(T); pred_h,proba_h = aggregate(refit_hier,T)
pd.DataFrame(dict(id=te['index'],gold=yt,direct=pred_d,hierarchical=pred_h,prob_clear=proba_h[:,0],prob_ambivalent=proba_h[:,1],prob_nonreply=proba_h[:,2])).to_csv(out/'predictions_hierarchy.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(dict(row=val_i,title=tr.iloc[val_i].title.to_numpy(),gold_fine=fine[val_i],pred_fine=val_fine,gold_coarse=y[val_i],direct=val_direct,hierarchical=val_hier)).to_csv(out/'validation_hierarchy.csv',index=False,encoding='utf-8-sig')

result = dict(seed=42,C=0.1,class_weight='balanced',feature_dim=int(X.shape[1]),split='GroupShuffleSplit title 80/20',train_n=len(tr),test_n=len(te),fit_n=len(fit_i),val_n=len(val_i),fit_groups=int(tr.iloc[fit_i].title.nunique()),val_groups=int(tr.iloc[val_i].title.nunique()),group_overlap=len(set(tr.iloc[fit_i].title)&set(tr.iloc[val_i].title)),fine_test_available=int(te.evasion_label.fillna('').str.strip().ne('').sum()),mapping={k:COARSE[v] for k,v in MAP.items()},counts={k:int((tr.evasion_label==k).sum()) for k in FINE},consistency=float((tr.evasion_label.map(MAP).to_numpy()==y).mean()),old_mapping_consistency=float((tr.evasion_label.map({**MAP,'Implicit':0}).to_numpy()==y).mean()),group_validation_direct=metrics(y[val_i],val_direct,COARSE),group_validation_hierarchy=metrics(y[val_i],val_hier,COARSE),group_validation_fine=metrics(fine[val_i],val_fine,FINE),official_direct=metrics(yt,pred_d,COARSE),official_hierarchy=metrics(yt,pred_h,COARSE))
result['source_audit'] = {k: {'train_unique':int(tr[k].nunique()),'test_unique':int(te[k].nunique()),'overlap_unique':len(set(tr[k])&set(te[k]))} for k in ['title','president','interview_answer','question']}
oldpred = pd.read_csv(base/'outputs'/'predictions_st_lr.csv')
assert oldpred.question.tolist()==te.question.tolist()
result['group_metrics'] = {}
for name,g in te.groupby('president'):
    ix=g.index.to_numpy(); result['group_metrics'][str(name)]={'n':len(ix),**metrics(yt[ix],pred_d[ix],COARSE)}
result['length_metrics']={}
length=te.interview_answer.str.split().str.len()
for name, mask in [('<=256 words',length<=256),('>256 words',length>256)]:
    ix=np.flatnonzero(mask); result['length_metrics'][name]={'n':len(ix),**metrics(yt[ix],pred_d[ix],COARSE)}
result['hashes']={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'outputs'/'st_embeddings.npz',base/'outputs'/'metrics_st_lr.json']}
(out/'metrics_hierarchy.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['group_metrics','length_metrics']},ensure_ascii=False,indent=2))
print('GROUPS',json.dumps({k: {'n':v['n'],'accuracy':v['accuracy'],'macro_f1':v['macro_f1']} for k,v in result['group_metrics'].items()}))
print('LENGTHS',json.dumps({k: {'n':v['n'],'accuracy':v['accuracy'],'macro_f1':v['macro_f1']} for k,v in result['length_metrics'].items()}))
print('ERROR SAMPLES')
errors=oldpred[oldpred.gold!=oldpred.prediction].copy()
for direction,g in errors.groupby(['gold','prediction']):
    print('DIRECTION',direction)
    g=g.assign(words=g.answer.str.split().str.len()).sort_values('words')
    for _,row in g.head(2).iterrows(): print(json.dumps(row.to_dict(),ensure_ascii=False))
