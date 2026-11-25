"""Confirm the datasets and PyArrow loaders agree for the pinned QEvasion Parquet files."""
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from pyarrow import parquet

paths={split:f'data/raw/qevasion-{split}.parquet' for split in ('train','test')}
loaded=load_dataset('parquet',data_files=paths,cache_dir=Path('data')/'.hf-data-audit')
counts={}
for split,dataset in loaded.items():
    path=Path(paths[split]); table=parquet.read_table(path).to_pylist()
    assert len(dataset)==len(table) and all(dataset[i]==row for i,row in enumerate(table))
    counts[split]={'rows':len(dataset),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rowwise_equal_to_pyarrow':True}
counts['loader']='datasets.load_dataset("parquet", data_files=verified QEvasion revision files)'
Path('evidence/dataset_loader_check.json').write_text(json.dumps(counts,indent=2)+'\n',encoding='utf-8')
print(counts)
