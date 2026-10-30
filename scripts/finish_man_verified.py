import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path('/data/EM-Benchmark')
OUT=ROOT/'runs/method_a/man_m3'
expected_counts=[None]*10
rows=[]
for i in range(1,11):
 p=OUT/f'step_{i:04d}'/'step.json'
 assert p.is_file(),f'Missing completed training step {i}; refusing test'
 d=json.loads(p.read_text());assert d['step']==i and d['epoch']==(1 if i<=5 else 2),d
 rows.append(d)
assert sum(d['train_episodes'] for d in rows)==388,'Training rollout coverage not 194 x 2'
for p in OUT.glob('step_*/gate/gate.json'):
 g=json.loads(p.read_text())
 for key in ('current','candidate'):
  assert g[key]['total']==34, f'{p} {key} incomplete validation denominator'
print('VERIFIED: 10 steps / 2 epochs / 388 rollout results / complete gate denominators',flush=True)
best=OUT/'skill/best.md'
method='method_a' if best.is_file() and best.read_text().strip() else 'baseline'
# baseline here is solely empty-skill Method A equivalence, never a second arm.
from em_skillopt.cli.evaluate import build_env
from em_skillopt.data.loaders import load_specs
manifest={}
for name in ['base','common_sense','complex','spatial','visual']:
 out=ROOT/'artifacts/man_method_a'/name
 env=build_env('eb-man',name,out,image_dir=out/'enumeration_images')
 try: specs=load_specs('eb-man',env,eval_set=name,split='test',limit=0)
 finally: env.close()
 expected={s.index for s in specs};manifest[name]=len(expected)
 cmd=[sys.executable,'-u','-m','em_skillopt.cli.evaluate','--env','eb-man','--method',method,
      '--eval-set',name,'--split','test','--limit','0','--parallel-workers','2',
      '--model','qwen3.8-27b','--proxy','','--n-shots','4','--max-tokens','8192','--out',str(out)]
 if method=='method_a':cmd+=['--skill-file',str(best)]
 for attempt in range(3):
  with (ROOT/'logs'/f'man_method_a_{name}.log').open('a') as log:
   rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
  saved=[json.loads(p.read_text()) for p in (out/'episodes').glob('*.json')]
  ids={r['spec']['index'] for r in saved}
  if ids==expected and len(saved)==len(expected):break
  print(f'INCOMPLETE test {name}: rc={rc} missing={sorted(expected-ids)}',flush=True)
 else:raise RuntimeError(f'{name} incomplete after three infrastructure recovery passes')
 print(f'VERIFIED test {name}: {len(saved)}/{len(expected)}',flush=True)
assert sum(manifest.values())==228,manifest
(ROOT/'artifacts/man_method_a/verified_manifest.json').write_text(json.dumps(manifest,indent=2))
print('MAN COMPLETE: 2 epochs and 228 persisted test episodes',flush=True)
