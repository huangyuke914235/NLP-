"""Train a joint MiniLM pair classifier without changing the fixed evaluation sets."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from run_experiments import Experiment, index_split, read_data, score, write_json

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(exp, family, data_file, labels, options, device, local):
    import torch
    frame=read_data(data_file,labels)
    encoded_labels=torch.tensor([labels.index(item) for item in frame['label']],dtype=torch.long)
    seq1=frame['question'].tolist()
    seq2=frame['answer'].tolist()
    train_idx,val_idx,test_idx=(index_split(frame,s) for s in ('train','validation','test'))
    revision=options['revision']
    checkpoint=options['model']
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,revision=revision,local_files_only=local)
    torch.manual_seed(options['seed'])
    model=AutoModelForSequenceClassification.from_pretrained(
        checkpoint,revision=revision,local_files_only=local,num_labels=len(labels),attn_implementation='eager',
    ).to(device)
    counts=torch.bincount(encoded_labels[train_idx],minlength=len(labels)).float()
    weight=counts.rsqrt() if family=='grouped' else torch.ones_like(counts)
    weight=(weight/weight.mean()).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    epochs=options['epochs'][family]
    batch_size=options['batch_size'] if device=='cuda' else 12
    total_steps=epochs*math.ceil(len(train_idx)/batch_size)
    warmup=max(1,round(options['warmup_fraction']*total_steps))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:(step+1)/warmup if step<warmup else max(.08,(total_steps-step)/(total_steps-warmup)))
    use_bfloat=device=='cuda' and torch.cuda.is_bf16_supported()
    random_draw=np.random.default_rng(options['seed'])
    history=[]
    best_score=-1.0
    best_state=None

    def batch(ids):
        result=tokenizer([seq1[i] for i in ids],[seq2[i] for i in ids],padding=True,
                         max_length=options['max_wordpieces'],truncation='only_second',return_tensors='pt')
        return {k:v.to(device) for k,v in result.items()}

    @torch.inference_mode()
    def predict(indices):
        model.eval()
        output=[]
        for start in range(0,len(indices),batch_size):
            part=indices[start:start+batch_size]
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=use_bfloat):
                logits=model(**batch(part)).logits
            output.append(logits.float().softmax(-1).cpu().numpy())
        return np.concatenate(output)

    for epoch in range(1,epochs+1):
        model.train()
        order=random_draw.permutation(train_idx)
        losses=[]
        for start in range(0,len(order),batch_size):
            ids=order[start:start+batch_size]
            gold=encoded_labels[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=use_bfloat):
                logits=model(**batch(ids)).logits
                loss=torch.nn.functional.cross_entropy(logits.float(),gold,weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            schedule.step()
            losses.append(float(loss.detach()))
        validation_p=predict(val_idx)
        valid=score(frame.iloc[val_idx]['label'],np.asarray(labels)[validation_p.argmax(axis=1)],labels)
        record={'epoch':epoch,'mean_batch_loss':float(np.mean(losses)),
                'validation_macro_f1':valid['macro_f1'],'validation_accuracy':valid['accuracy'],
                'validation_per_class':valid['per_class']}
        history.append(record)
        exp.log(f'{family} epoch={epoch} train_loss={record["mean_batch_loss"]:.4f} validation_macro_f1={valid["macro_f1"]:.6f} validation_accuracy={valid["accuracy"]:.6f}')
        if valid['macro_f1']>best_score:
            best_score=valid['macro_f1']
            best_record=record
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    exp.add_selection(f'{family}_pair',history,best_record)
    parameters={'best_epoch':best_record['epoch'],'validation_macro_f1':best_score,
                'learning_rate':options['learning_rate'],'batch_size':batch_size,
                'train_count':len(train_idx),'validation_count':len(val_idx),
                'model_parameters':sum(v.numel() for v in model.parameters()),'used_bfloat16':use_bfloat,
                'class_weights':dict(zip(labels,weight.cpu().tolist())), 'max_wordpieces':options['max_wordpieces']}
    exp.evaluate(f'{family}_pair',frame.iloc[test_idx],frame['label'].to_numpy()[test_idx],
                 predict(test_idx),labels,parameters)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/processed')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    opts_path=ROOT/'configs/pair_finetuning.json'
    options=json.loads(opts_path.read_text(encoding='utf-8'))
    base=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    random.seed(options['seed']);np.random.seed(options['seed'])
    torch.set_num_threads(base['threads'])
    torch.use_deterministic_algorithms(True)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    exp=Experiment(args.output,base)
    start=perf_counter()
    try:
        weight_path=hf_hub_download(options['model'],'model.safetensors',revision=options['revision'],local_files_only=True)
        local=True
    except FileNotFoundError:
        weight_path=hf_hub_download(options['model'],'model.safetensors',revision=options['revision'])
        local=False
    exp.log(f'Joint encoder torch={torch.__version__}, device={device}, weights={sha(Path(weight_path))}')
    inputs={}
    for domain,filename,label_space in [('grouped','qevasion_grouped.csv','source_labels'),('target','qnli.csv','target_labels')]:
        path=args.data/filename
        inputs[filename]=sha(path)
        train(exp,domain,path,base[label_space],options,device,local)
    write_json(args.output/'environment.json',{
        'python':sys.version.split()[0],'torch':torch.__version__,'transformers':version('transformers'),'device':device,
        'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
        'code_sha256':sha(Path(__file__)),'options_sha256':sha(opts_path),'base_weights_sha256':sha(Path(weight_path)),
        'inputs_sha256':inputs,'elapsed_seconds':round(perf_counter()-start,3),
        'command':[sys.executable,*sys.argv],
    })
    exp.log('Completed')


if __name__=='__main__': main()
