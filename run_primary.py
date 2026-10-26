"""Primary benchmark: equal-budget tuning on development episodes, MPC mechanism factorial, and paired held-out evaluation."""
import os,sys,copy,json,hashlib,time,datetime,itertools,csv,argparse,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,os.environ.get('QUAD_BENCH_DEPS',str(ROOT.parent/'.deps')))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.stats import qmc
from simulation import simulate,augmented_metrics
BASE=json.loads((ROOT/'config.json').read_text())
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)

def digest(data):return hashlib.sha256(data).hexdigest()
def cfg_hash(c):return digest(json.dumps(c,sort_keys=True).encode())
def job(key,c,name,tr,case,group,**meta):return dict(id=key,config=c,controller=name,trajectory=tr,case=case,group=group,variant=meta.get('variant','default'),sample=meta.get('sample',0),**{k:v for k,v in meta.items() if k not in ['variant','sample']})
def new_config(split):
 c=copy.deepcopy(BASE);c['new_split']=split;c['MPC']['preview']=False;s=c['scenarios']
 s.update(mc_mass_bounds=[.8,1.25],mc_inertia_bounds=[.75,1.25],mc_thrust_bounds=[.8,1.15],wind_interval=[5.4,11.4],wind_force=[.7,-.45,.25])
 if split=='devA':c['seed']=310001;s.update(figure8_A=.8,figure8_B=.55,figure8_C=.1,figure8_omega=.42,figure8_vertical_omega=.31,path_phase=.25)
 elif split=='devB':c['seed']=310101;s.update(step_target=[1.4,-.8,1.2],step_time=2.4,smooth_duration=4.5)
 elif split=='validation':c['seed']=410001;s.update(figure8_A=1.1,figure8_B=.65,figure8_C=.15,figure8_omega=.38,figure8_vertical_omega=.29,path_phase=.8)
 else:c['seed']=510001
 return c

def candidates(name):
 points=qmc.LatinHypercube(d=4,seed=20261009+['PID','LQR','SMC','MPC'].index(name)).random(11);out=[copy.deepcopy(BASE[name])]
 for row in points:
  s=copy.deepcopy(BASE[name]);a,b,d,e=.6+1.2*row
  if name=='PID':
   s['kp']=(np.array(s['kp'])*a).tolist();s['kd']=(np.array(s['kd'])*b).tolist();s['ki']=(np.array(s['ki'])*d).tolist();s['att_kp']=(np.array(s['att_kp'])*e).tolist();s['att_kd']=(np.array(s['att_kd'])*np.sqrt(e)).tolist()
  elif name=='SMC':
   s['lambda_pos']=(np.array(s['lambda_pos'])*a).tolist();s['k_pos']=(np.array(s['k_pos'])*b).tolist();s['k_pos'][2]=float(1.5+3*row[1]);s['boundary_pos']=(np.array(s['boundary_pos'])*d).tolist();s['k_att']=(np.array(s['k_att'])*e).tolist()
  else:
   Q=np.array(s['Q'],dtype=float);Q[:3]*=a;Q[3:6]*=b;s['Q']=Q.tolist();s['R']=(np.array(s['R'])*d).tolist()
   if name=='MPC':s['S']=(np.array(s['S'])*e).tolist()
   else:Q[6:9]*=e;s['Q']=Q.tolist()
  if name=='MPC':s['preview']=False
  out.append(s)
 if name=='MPC':out[0]['preview']=False
 return out

def freeze():
 if (OUT/'protocol_frozen.json').exists():return json.loads((OUT/'protocol_frozen.json').read_text())
 protocol=dict(local_freeze_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),registration='Local timestamp/hash recorded before execution; not externally registered',candidate_budget=12,training_episodes_per_candidate=2,validation_top_k=3,selection_rule='Mean development objective ascending, tie candidate index; evaluate top 3 on separate validation episode and choose minimum validation objective, tie candidate index',objective='100*failure + RMSE/1m + 0.05*log(1+applied_variation/1)',failure_policy='Keep all failures and partial diagnostics; failure penalty dominates this candidate range, not formal safety lexicographic guarantee',heldout_pairs=50,primary_test_metric='Failure frequency plus completed-episode RMSE; no ranking by partial-horizon RMSE',bootstrap=dict(seed=920001,replicates=10000,unit='paired realization',interval='percentile 95%; complete-case paired RMSE/variation only with explicit pair count'),tuning_design='12 candidate settings in each declared four-dimensional space; baseline + 11 Latin-hypercube points, equal candidate/episode counts not equal compute; no common optimality guarantee',predictor='16-state hover linearization of motor-speed dynamics, same generalized command coordinates, ideal measured motor state available and used only by augmented predictor; zero motor-state stage Q, own model DARE terminal P',backup='Raw pinned rigid LQR versus weighted Euclidean projection into known rotor/increment set; projection failure holds prior request and is logged; no invariant/safety proof',tilt='hard_initial versus hard_future versus soft_future; soft penalty 10000 per squared rad, no slack upper bound; gate checks declared slack feasibility rather than original hard-tilt safety',mechanism='32 A factorial cells plus 16 non-overlapping B future-policy cells; deterministic diagnostic tests on already observed task family',latency='Exogenous replay of measured dedicated timing with command hold/discard at 20ms; within-sample old/new motor command integration; computation/commit atomic offline; no hardware/SIL claim',preview_policy='Fixed-gain and tuned MPC both run without reference preview on the held-out track',base_config_sha256=digest((ROOT/'config.json').read_bytes()))
 protocol['design_source_sha256']={n:digest((ROOT/n).read_bytes()) for n in ['controllers.py','base_controllers.py','plant.py','references.py','metrics.py','simulation.py','run_primary.py']}
 protocol['candidates']={n:candidates(n) for n in ['PID','LQR','SMC','MPC']};protocol['splits']={s:new_config(s) for s in ['devA','devB','validation','test']}
 raw=json.dumps(protocol,indent=2).encode();(OUT/'protocol_frozen.json').write_bytes(raw);(OUT/'protocol_frozen.sha256').write_text(digest(raw));print('PROTOCOL FROZEN',protocol['local_freeze_utc'],digest(raw),flush=True)
 return protocol

def execute(j):
 path=OUT/'raw'/(j['id']+'.npz');meta=path.with_suffix('.json')
 if path.exists():
  info=json.loads(meta.read_text());assert info['job']==j,'Configuration mismatch: '+j['id']
  z=np.load(__import__('io').BytesIO(path.read_bytes()));log={k:z[k] for k in z.files};z.close();actual=info['actual_plant'];elapsed=info['worker_wall_s']
 else:
  start=time.perf_counter();log,actual=simulate(j);elapsed=time.perf_counter()-start
  np.savez_compressed(path,**log);meta.write_text(json.dumps(dict(job=j,actual_plant=actual,worker_wall_s=elapsed),indent=2))
 m=augmented_metrics(log,j['config']);valid=np.isfinite(log['x'][:,0]);m.update(id=j['id'],group=j['group'],controller=j['controller'],trajectory=j['trajectory'],case=j['case'],variant=j['variant'],sample=j['sample'],worker_wall_s=elapsed,config_sha256=cfg_hash(j['config']),raw_sha256=digest(path.read_bytes()))
 m['objective']=100*m['failure']+m['rmse_3d']+.05*np.log1p(m['variation']);m['gate_rejected']=int(np.sum(np.isin(log['status_val'],[1,2])&~log['qp_accepted'])) if j['controller']=='MPC' else 0
 m['selected_slack_max']=float(np.nanmax(log['qp_slack_max'][log['qp_accepted']])) if j['controller']=='MPC' and log['qp_accepted'].any() else None
 m['prediction_position_mean']=float(np.nanmean(log['prediction_position_error'][valid])) if j['controller']=='MPC' else None
 for key in ['track','candidate_index','split','preview','predictor','backup_policy','tilt_policy']:m[key]=j.get(key,'')
 (OUT/'raw'/(j['id']+'_metrics.json')).write_text(json.dumps(m,indent=2));return m

def batch(js,label):
 (OUT/(label+'_plan.json')).write_text(json.dumps(js,indent=2))
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(execute,j):j['id'] for j in js}
  for fut in concurrent.futures.as_completed(futures):
   m=fut.result();rows.append(m);print(label,len(rows),'/',len(js),m['id'],'F',m['failure'],'RMSE',round(m['rmse_3d'],5),flush=True)
 rows.sort(key=lambda m:m['id']);(OUT/(label+'_metrics.json')).write_text(json.dumps(rows,indent=2));return rows

def train(p):
 js=[]
 for name,settings in p['candidates'].items():
  for i,s in enumerate(settings):
   for split,tr in [('devA','novel8'),('devB','smooth')]:
    c=copy.deepcopy(p['splits'][split]);c[name]=copy.deepcopy(s);js.append(job(f'train_{name}_{i:02d}_{split}',c,name,tr,'MC','train',candidate_index=i,split=split))
 rows=batch(js,'training');tops={}
 for name in p['candidates']:
  scores=[(np.mean([m['objective'] for m in rows if m['controller']==name and m['candidate_index']==i]),i) for i in range(12)];tops[name]=[i for score,i in sorted(scores)[:3]]
 js=[]
 for name,inds in tops.items():
  for i in inds:
   c=copy.deepcopy(p['splits']['validation']);c[name]=copy.deepcopy(p['candidates'][name][i]);js.append(job(f'valid_{name}_{i:02d}',c,name,'novel8','MC','validation',candidate_index=i,split='validation'))
 val=batch(js,'validation');selection={}
 for name in p['candidates']:
  best=min([m for m in val if m['controller']==name],key=lambda m:(m['objective'],m['candidate_index']));i=best['candidate_index'];selection[name]=dict(candidate_index=i,settings=p['candidates'][name][i],validation_objective=best['objective'],development_top3=tops[name])
 payload=json.dumps(selection,indent=2).encode();(OUT/'selection_locked.json').write_bytes(payload);(OUT/'selection_locked.sha256').write_text(digest(payload));print('SELECTION LOCKED',digest(payload),flush=True)

def mechanisms(p):
 js=[]
 for pr,model,backup,case,tr in itertools.product([False,True],['rigid','motor'],['raw','projected'],['nominal','C1a'],['step','smooth']):
  c=copy.deepcopy(BASE);c['MPC'].update(preview=pr,predictor=model,backup_policy=backup,tilt_policy='hard_initial')
  key=f'A_{int(pr)}_{model}_{backup}_{case}_{tr}';js.append(job(key,c,'MPC',tr,case,'mechanism',preview=pr,predictor=model,backup_policy=backup,tilt_policy='hard_initial'))
 for policy,pr,case,tr in itertools.product(['hard_future','soft_future'],[False,True],['nominal','C1a'],['step','smooth']):
  c=copy.deepcopy(BASE);c['MPC'].update(preview=pr,predictor='rigid',backup_policy='raw',tilt_policy=policy,slack_penalty=10000.)
  key=f'B_{policy}_{int(pr)}_{case}_{tr}';js.append(job(key,c,'MPC',tr,case,'tilt',preview=pr,predictor='rigid',backup_policy='raw',tilt_policy=policy))
 return batch(js,'mechanism')

def test(p):
 selection=json.loads((OUT/'selection_locked.json').read_text());js=[]
 for sample in range(50):
  rng=np.random.default_rng(610001+sample);c=copy.deepcopy(p['splits']['test']);s=c['scenarios'];s.update(figure8_A=float(rng.uniform(.8,1.25)),figure8_B=float(rng.uniform(.45,.8)),figure8_C=float(rng.uniform(.1,.2)),figure8_omega=float(rng.uniform(.4,.55)),figure8_vertical_omega=float(rng.uniform(.3,.4)),path_phase=float(rng.uniform(0,1.2)))
  for track,name in itertools.product(['fixed','tuned'],['PID','LQR','SMC','MPC']):
   cc=copy.deepcopy(c)
   if track=='tuned':cc[name]=copy.deepcopy(selection[name]['settings'])
   js.append(job(f'test_{track}_{name}_{sample:02d}',cc,name,'novel8','MC','test',sample=sample,track=track,split='test'))
 return batch(js,'heldout')

def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['freeze','train','mechanism','test','all'],default='all');arg=a.parse_args();p=freeze();assert digest((OUT/'protocol_frozen.json').read_bytes())==(OUT/'protocol_frozen.sha256').read_text().strip()
 if arg.stage in ['train','all']:train(p)
 if arg.stage in ['mechanism','all']:mechanisms(p)
 if arg.stage in ['test','all']:test(p)
if __name__=='__main__':main()

