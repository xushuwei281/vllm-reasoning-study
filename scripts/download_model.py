"""Download the pinned reference checkpoint into explicitly configured HF_HOME."""
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
if not os.environ.get('HF_HOME'):raise SystemExit('Set HF_HOME to the study scratch cache before downloading.')
path=snapshot_download('Qwen/Qwen3-1.7B',revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',max_workers=2)
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(path+'\n')
print(path)
