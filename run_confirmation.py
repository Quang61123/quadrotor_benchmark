"""Confirmation benchmark: fresh evaluation realizations, preview-enabled MPC (MPCP), geometric SO(3) baseline (GEO),
and nested tuning budgets of 12/24/48 candidates. Requires outputs/protocol_frozen.json from run_primary.py.
Raw episodes go to QUAD_CONFIRMATION_RAW (default outputs/confirmation/raw); metrics to outputs/confirmation/."""
import os,sys,copy,json,hashlib,time,datetime,itertools,argparse,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,os.environ.get('QUAD_BENCH_DEPS',str(ROOT.parent/'.deps')))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.stats import qmc
import simulation,geometric_controller
simulation.Controller=geometric_controller.Controller
import run_primary as RS
PRIMARY_OUT=ROOT/'outputs';OUT=PRIMARY_OUT/'confirmation';OUT.mkdir(parents=True,exist_ok=True)
RAW=Path(os.environ.get('QUAD_CONFIRMATION_RAW',str(OUT/'raw')));RAW.mkdir(parents=True,exist_ok=True)
RS.OUT=RAW.parent
# execute() writes to RS.OUT/'raw'; make RAW exactly that folder
assert (RS.OUT/'raw')==RAW,'QUAD_CONFIRMATION_RAW must end with a folder named raw'
BASE=RS.BASE
BASE.setdefault('GEO',dict(kx=[2.5,2.5,4],kv=[3,3,3.5],kR=[64,64,36],kW=[16,16,12],tilt_max=.55))
CTRL=['PID','LQR','SMC','MPC','GEO','MPCP']
BUDGETS=[12,24,48]
def base_settings(name):
 s=copy.deepcopy(BASE['MPC' if name=='MPCP' else name])
 if name in('MPC','MPCP'):s['preview']=(name=='MPCP')
 return s
def map_row(name,row):
 s=base_settings(name);a,b,d,e=.6+1.2*row
 if name=='PID':
  s['kp']=(np.array(s['kp'])*a).tolist();s['kd']=(np.array(s['kd'])*b).tolist();s['ki']=(np.array(s['ki'])*d).tolist();s['att_kp']=(np.array(s['att_kp'])*e).tolist();s['att_kd']=(np.array(s['att_kd'])*np.sqrt(e)).tolist()
 elif name=='SMC':
  s['lambda_pos']=(np.array(s['lambda_pos'])*a).tolist();s['k_pos']=(np.array(s['k_pos'])*b).tolist();s['k_pos'][2]=float(1.5+3*row[1]);s['boundary_pos']=(np.array(s['boundary_pos'])*d).tolist();s['k_att']=(np.array(s['k_att'])*e).tolist()
 elif name=='GEO':
  s['kx']=(np.array(s['kx'])*a).tolist();s['kv']=(np.array(s['kv'])*b).tolist();s['kR']=(np.array(s['kR'])*d).tolist();s['kW']=(np.array(s['kW'])*e).tolist()
 else:
  Q=np.array(s['Q'],dtype=float);Q[:3]*=a;Q[3:6]*=b;s['Q']=Q.tolist();s['R']=(np.array(s['R'])*d).tolist()
  if name in('MPC','MPCP'):s['S']=(np.array(s['S'])*e).tolist()
  else:Q[6:9]*=e;s['Q']=Q.tolist()
 return s
def candidates(name):
 # PID/LQR/SMC/MPC reuse the 12 primary-benchmark candidates; GEO gets its own baseline + 11 LHS points
 if name in['PID','LQR','SMC','MPC']:first=RS.candidates(name)
 else:
  first=[base_settings(name)]+[map_row(name,r) for r in qmc.LatinHypercube(d=4,seed=20261009+(4 if name=='GEO' else 3)).random(11)]
 if name=='MPCP':first=[dict(c,preview=True) for c in RS.candidates('MPC')]
 extra=qmc.LatinHypercube(d=4,seed=20261019+CTRL.index(name)).random(36)
 return first+[map_row(name,r) for r in extra]
def protocol():
 f=OUT/'protocol_confirmation.json'
 if f.exists():return json.loads(f.read_text())
 primary=json.loads((PRIMARY_OUT/'protocol_frozen.json').read_text())
 splits=copy.deepcopy(primary['splits'])
 for k in splits:splits[k]['GEO']=copy.deepcopy(BASE['GEO'])
 fresh=copy.deepcopy(splits['test']);fresh['seed']=520001;splits['fresh']=fresh
 p=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Fresh 50-realization evaluation; preview-enabled MPC track (MPCP) tuned with the same protocol; geometric SO(3) baseline (GEO); nested tuning budgets 12/24/48 (first 12 candidates identical to the primary benchmark for PID/LQR/SMC/MPC)',fresh_realization_rng_seed_base=810001,fresh_plant_seed=520001,budgets=BUDGETS,selection_rule=primary['selection_rule'],objective=primary['objective'],candidates={n:candidates(n) for n in CTRL},splits=splits,source_sha256={n:RS.digest((ROOT/n).read_bytes()) for n in ['controllers.py','geometric_controller.py','base_controllers.py','plant.py','references.py','metrics.py','simulation.py','run_primary.py','run_confirmation.py']})
 raw=json.dumps(p,indent=2).encode();f.write_bytes(raw);(OUT/'protocol_confirmation.sha256').write_text(RS.digest(raw));return p
def jobfor(key,c,name,tr,group,**meta):
 cc=copy.deepcopy(c);ctl='MPC' if name=='MPCP' else name
 return RS.job(key,cc,ctl,tr,'MC',group,method=name,**meta)
def setcfg(c,name,s):
 c=copy.deepcopy(c);c['MPC' if name=='MPCP' else name]=copy.deepcopy(s);return c
def batch(js,label,workers):
 done=[];todo=[]
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
  fut={pool.submit(RS.execute,j):j for j in js}
  for f in concurrent.futures.as_completed(fut):
   m=f.result();m['method']=fut[f]['method'];rows.append(m)
   if len(rows)%10==0 or len(rows)==len(js):print(label,len(rows),'/',len(js),time.strftime('%H:%M:%S'),flush=True)
 rows.sort(key=lambda m:m['id']);(OUT/(label+'_metrics.json')).write_text(json.dumps(rows,indent=2,default=float));return rows
def tune(p,workers):
 js=[]
 for name in CTRL:
  for i,s in enumerate(p['candidates'][name]):
   for split,tr in [('devA','novel8'),('devB','smooth')]:
    js.append(jobfor(f'train_{name}_{i:02d}_{split}',setcfg(p['splits'][split],name,s),name,tr,'train',candidate_index=i,split=split))
 rows=batch(js,'training',workers);need=set();tops={}
 for name in CTRL:
  for B in BUDGETS:
   sc=sorted((np.mean([m['objective'] for m in rows if m['method']==name and m['candidate_index']==i]),i) for i in range(B));tops[(name,B)]=[i for _,i in sc[:3]];need|={(name,i) for i in tops[(name,B)]}
 js=[jobfor(f'valid_{n}_{i:02d}',setcfg(p['splits']['validation'],n,p['candidates'][n][i]),n,'novel8','validation',candidate_index=i,split='validation') for n,i in sorted(need)]
 val=batch(js,'validation',workers);sel={}
 for name in CTRL:
  for B in BUDGETS:
   best=min([m for m in val if m['method']==name and m['candidate_index'] in tops[(name,B)]],key=lambda m:(m['objective'],m['candidate_index']))
   sel[f'{name}@{B}']=dict(method=name,budget=B,candidate_index=best['candidate_index'],validation_objective=best['objective'],development_top3=tops[(name,B)],settings=p['candidates'][name][best['candidate_index']])
 raw=json.dumps(sel,indent=2).encode();(OUT/'selection.json').write_bytes(raw);(OUT/'selection.sha256').write_text(RS.digest(raw));print('SELECTION LOCKED',RS.digest(raw),flush=True)
def fresh(p,workers):
 sel=json.loads((OUT/'selection.json').read_text());js=[]
 for sample in range(50):
  rng=np.random.default_rng(p['fresh_realization_rng_seed_base']+sample);c=copy.deepcopy(p['splits']['fresh']);s=c['scenarios'];s.update(figure8_A=float(rng.uniform(.8,1.25)),figure8_B=float(rng.uniform(.45,.8)),figure8_C=float(rng.uniform(.1,.2)),figure8_omega=float(rng.uniform(.4,.55)),figure8_vertical_omega=float(rng.uniform(.3,.4)),path_phase=float(rng.uniform(0,1.2)))
  for name in CTRL:
   js.append(jobfor(f'fresh_fixed_{name}_{sample:02d}',setcfg(c,name,p['candidates'][name][0]),name,'novel8','fresh',sample=sample,track='fixed',split='fresh'))
   seen={0:'fixed'}
   for B in BUDGETS:
    i=sel[f'{name}@{B}']['candidate_index']
    if i in seen:continue
    seen[i]=B;js.append(jobfor(f'fresh_cand{i:02d}_{name}_{sample:02d}',setcfg(c,name,p['candidates'][name][i]),name,'novel8','fresh',sample=sample,track=f'cand{i:02d}',split='fresh',candidate_index=i))
 batch(js,'fresh',workers)
def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['protocol','tune','fresh','all'],default='all');a.add_argument('--workers',type=int,default=5);g=a.parse_args()
 p=protocol();print('PROTOCOL',(OUT/'protocol_confirmation.sha256').read_text(),flush=True)
 if g.stage in('tune','all'):tune(p,g.workers)
 if g.stage in('fresh','all'):fresh(p,g.workers)
 print('DONE',time.strftime('%H:%M:%S'),flush=True)
if __name__=='__main__':main()
