"""Source-only QNLI transfer with an explicit, fixed coarse-label projection."""
import argparse, json, sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--output',type=Path)
a=parser.parse_args(); base=a.project.resolve(); out=a.output or base/'outputs'/'hierarchy'; out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(base/'deps'))
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
train=load_from_disk(str(base/'data'/'qevasion_raw'))['train'].to_pandas()
q=pd.read_csv(base/'outputs'/'predictions_qnli_tfidf_lr.csv')
model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.95)),('lr',LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=42))])
y=train.clarity_label.map({'Clear Reply':0,'Ambivalent':1,'Clear Non-Reply':2})
model.fit(train.question+' [SEP] '+train.interview_answer,y)
p=model.predict_proba(q.question+' [SEP] '+q.answer)
binary=np.column_stack([p[:,0],p[:,1]+p[:,2]])
pred=binary.argmax(axis=1); gold=q.gold.map({'Entailment (contains answer)':0,'Not Entailment':1}).to_numpy()
assert not pd.isna(gold).any()
def metric(pred):
    return dict(accuracy=float(accuracy_score(gold,pred)),macro_f1=float(f1_score(gold,pred,average='macro')),per_class=classification_report(gold,pred,output_dict=True,zero_division=0),confusion_matrix=confusion_matrix(gold,pred).tolist())
result={'source_only_projection':metric(pred),'target_train_tie_majority':metric(np.zeros_like(gold)),'projection':'P(entailment)=P(Clear Reply); P(not_entailment)=P(Ambivalent)+P(Clear Non-Reply)','target_labels_used_for_training':False,'threshold':0.5,'seed':42,'source_train_n':len(train),'qnli_dev_n':len(q)}
(out/'metrics_direct_transfer.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(dict(id=q.id,gold=gold,prediction=pred,p_entailment=binary[:,0],p_not_entailment=binary[:,1])).to_csv(out/'predictions_direct_transfer.csv',index=False,encoding='utf-8-sig')
print(json.dumps(result,ensure_ascii=False,indent=2))
