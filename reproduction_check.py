"""Re-execute four held-out episodes of the primary benchmark and compare them with the stored metrics.
Reads outputs/protocol_frozen.json, outputs/selection_locked.json and outputs/heldout_metrics.json;
writes outputs/reproduction_check.json."""
import os,sys,copy,json,itertools
from pathlib import Path
ROOT=Path(__file__).parent;sys.path.insert(0,os.environ.get('QUAD_BENCH_DEPS',str(ROOT.parent/'.deps')))
import numpy as np,simulation
OUT=ROOT/'outputs'
protocol=json.loads((OUT/'protocol_frozen.json').read_text());sel=json.loads((OUT/'selection_locked.json').read_text());ref={m['id']:m for m in json.loads((OUT/'heldout_metrics.json').read_text())}
out=[]
for sample,(track,name) in itertools.product([0,7],[('fixed','PID'),('tuned','MPC')]):
 rng=np.random.default_rng(610001+sample);c=copy.deepcopy(protocol['splits']['test']);s=c['scenarios'];s.update(figure8_A=float(rng.uniform(.8,1.25)),figure8_B=float(rng.uniform(.45,.8)),figure8_C=float(rng.uniform(.1,.2)),figure8_omega=float(rng.uniform(.4,.55)),figure8_vertical_omega=float(rng.uniform(.3,.4)),path_phase=float(rng.uniform(0,1.2)))
 if track=='tuned':c[name]=copy.deepcopy(sel[name]['settings'])
 j=dict(id=f'test_{track}_{name}_{sample:02d}',config=c,controller=name,trajectory='novel8',case='MC',group='test',variant='default',sample=sample,track=track,split='test')
 log,_=simulation.simulate(j);m=simulation.augmented_metrics(log,c);r=ref[j['id']]
 out.append(dict(id=j['id'],rmse_now=m['rmse_3d'],rmse_stored=r['rmse_3d'],var_now=m['variation'],var_stored=r['variation'],abs_diff=abs(m['rmse_3d']-r['rmse_3d'])))
print(json.dumps(out,indent=1));(OUT/'reproduction_check.json').write_text(json.dumps(out,indent=1))
