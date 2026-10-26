"""Closed-loop episode simulation, episode metrics, and a scenario sweep (python simulation.py)."""
import os,sys,json,copy,time,csv,hashlib,platform,itertools
from pathlib import Path
ROOT=Path(__file__).parent
if os.environ.get('QUAD_BENCH_USE_ENV')!='1':sys.path.insert(0,os.environ.get('QUAD_BENCH_DEPS',str(ROOT.parent/'.deps')))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np,scipy,osqp
from plant import mix,integrate,allocate
from references import ref,disturbance
from controllers import Controller
from metrics import metrics
LABELS={'PID':'PID','LQR':'hover LQR','SMC':'boundary-layer SMC','MPC':'LMPC + LQR fallback'}
def jobs(c):
 out=[]
 def add(group,tr,case,name,variant='baseline',sample=0,changes=None):
  cfg=copy.deepcopy(c)
  for section,values in (changes or {}).items():cfg.setdefault(section,{}).update(values)
  out.append(dict(id=f'{group}_{tr}_{case}_{sample:02d}_{name}_{variant}',group=group,trajectory=tr,case=case,controller=name,variant=variant,sample=sample,config=cfg))
 for tr,case,name in itertools.product(['hover','step','circle','figure8'],['nominal','D1','D2','D3','U1','U2','C1a','C1b','C2'],LABELS):add('core',tr,case,name)
 for sample,name in itertools.product(range(c['mc_samples']),LABELS):add('core','figure8','MC',name,sample=sample)
 add('preview','step','nominal','MPC','no_preview',changes={'MPC':{'preview':False}})
 for name in LABELS:add('smooth','smooth','nominal',name)
 for case,name in itertools.product(['C1a','C1b','C2'],LABELS):add('stress','step',case,name)
 for case,name in itertools.product(['C1a','C1b'],LABELS):add('smooth_stress','smooth',case,name)
 for tau in [.005,.035]:add('lag_surrogate','smooth','nominal','MPC',f'tau_{tau}',changes={'plant':{'tau_m':tau}})
 for gain,case,tr in itertools.product(c['sensitivity']['smc_vertical_gains'],['U1','U2'],['hover','figure8']):
  kp=c['SMC']['k_pos'].copy();kp[2]=gain;add('smc_gain',tr,case,'SMC',f'kz_{gain}',changes={'SMC':{'k_pos':kp}})
 for scale in c['sensitivity']['smc_boundary_scales']:
  add('smc_boundary','figure8','D1','SMC',f'layer_{scale}',changes={'SMC':{'boundary_pos':(np.array(c['SMC']['boundary_pos'])*scale).tolist()}})
 for horizon in c['sensitivity']['mpc_horizons']:add('mpc_horizon','circle','nominal','MPC',f'N_{horizon}',changes={'MPC':{'N':horizon}})
 for scale in c['sensitivity']['mpc_rate_scales']:add('mpc_rate','circle','nominal','MPC',f'S_{scale}',changes={'MPC':{'S':(np.array(c['MPC']['S'])*scale).tolist()}})
 for pattern,name in itertools.product(['periodic','burst'],LABELS):add('delay','step','nominal',name,pattern)
 for i,(m,j,kf) in enumerate(itertools.product(c['corner']['mass'],c['corner']['inertia'],c['corner']['thrust'])):
  for name in LABELS:add('corner','figure8','corner',name,f'm{m}_J{j}_kf{kf}',sample=i,changes={'corner_actual':{'m':m,'J':j,'kf':kf}})
 for case in ['nominal','C1a']:
  for horizon in c['sensitivity']['mpc_horizons']:add('hard_horizon','step',case,'MPC',f'N_{horizon}',changes={'MPC':{'N':horizon,'preview':False}})
  for scale in c['sensitivity']['mpc_rate_scales']:add('hard_rate','step',case,'MPC',f'S_{scale}',changes={'MPC':{'S':(np.array(c['MPC']['S'])*scale).tolist(),'preview':False}})
  for tau in [.005,.035]:add('hard_lag','step',case,'MPC',f'tau_{tau}',changes={'plant':{'tau_m':tau},'MPC':{'preview':False}})
 return out
def simulate(job):
 c=job['config'];name=job['controller'];tr=job['trajectory'];case=job['case'];dt=c['dt'];p=copy.deepcopy(c['plant']);alloc=copy.deepcopy(p);cc=copy.deepcopy(c);seed=c['seed']+job['sample']
 if case in ('U1','U2'):p['m']*=1.2
 if case=='U2':p['J']=(np.array(p['J'])*[1.15,.85,1.1]).tolist();p['kf']*=.85
 if case=='MC':
  rng=np.random.default_rng(seed);s=c['scenarios'];p['m']*=rng.uniform(*s['mc_mass_bounds']);p['J']=(np.array(p['J'])*rng.uniform(*s['mc_inertia_bounds'],size=3)).tolist();p['kf']*=rng.uniform(*s['mc_thrust_bounds'])
 if case=='corner':
  q=c['corner_actual'];p['m']*=q['m'];p['J']=(np.array(p['J'])*q['J']).tolist();p['kf']*=q['kf']
 if case in ('C1a','C1b'):
  p['omega_max']=alloc['omega_max']=c['scenarios']['C1_omega_max']
  if case=='C1a':cc['plant']['omega_max']=p['omega_max']
 slew=c['scenarios']['C2_speed_slew'] if case=='C2' else np.inf
 ctl=Controller(name,cc,tr);ts=np.arange(0,c['duration'],dt);n=len(ts);force=disturbance(ts,case,c,seed);x=np.zeros(12);x[:3],x[3:6],_=ref(0,tr,c)
 if tr=='hover':x[2]=.7
 om=np.full(4,np.sqrt(c['plant']['m']*c['plant']['g']/4/c['plant']['kf']));cmd=om.copy();uaccepted=np.array([c['plant']['m']*c['plant']['g'],0,0,0]);M=mix(p);Wa=np.linalg.inv(mix(alloc))
 log={'t':ts,'x':np.full((n,12),np.nan),'ref':np.full((n,3),np.nan),'force':force}
 for key in ['u','requested','candidate','motors','commands','rotor_requested','rotor_allocated','speed_requested','previous_rate_input','qp_candidate','backup_command']:log[key]=np.full((n,4),np.nan)
 for key in ['smc_surface','smc_switch']:log[key]=np.full((n,3),np.nan)
 for key in ['qp_slack_max','qp_initial_defect','qp_dynamics_defect','projection_norm','prediction_position_error','prediction_velocity_error','time_ms','primal_residual','dual_residual','solve_time_ms','objective','synthetic_delay_ms','qp_rotor_violation_N','qp_tilt_violation_rad','qp_increment_violation_0','qp_increment_violation_1','qp_increment_violation_2','qp_increment_violation_3']:log[key]=np.full(n,np.nan)
 for key in ['projection_bad','sat','rate','solver_bad','motor_violation','current_attitude_outside_qp','warm_start_enabled','warm_start_previous','late','qp_accepted','fallback','applied_fallback']:log[key]=np.zeros(n,bool)
 for key in ['solver_iter','status_val']:log[key]=np.zeros(n,int)
 log['solver_status']=np.full(n,'not_applicable',dtype='U40');log['fallback_reason']=np.full(n,'',dtype='U60')
 log['qp_vector']=np.full((n,ctl.nxv+cc['MPC']['N']*4+ctl.ns if name=='MPC' else 0),np.nan)
 accepted_fallback=False
 failed=False;fail_time=np.nan;fail_reason='';fail_state=np.full(12,np.nan)
 for k,t in enumerate(ts):
  log['x'][k]=x;log['ref'][k]=ref(t,tr,c)[0];log['u'][k]=M@(p['kf']*om**2);log['motors'][k]=om;log['previous_rate_input'][k]=ctl.prev
  ctl.motor_state=om.copy()
  tick=time.perf_counter_ns();candidate=ctl.command(t,x);log['time_ms'][k]=(time.perf_counter_ns()-tick)/1e6;u=candidate;applied_fallback=bool(ctl.diag.get('fallback',False))
  if name=='MPC':
   log['qp_candidate'][k]=ctl.qp_candidate;log['backup_command'][k]=ctl.backup_command
   if ctl.qp_vector is not None:log['qp_vector'][k]=ctl.qp_vector
  if job['group']=='delay':
   q=c['delay'];late=(k%q['periodic_every']==0 and k>0) if job['variant']=='periodic' else q['burst_start']<=t<q['burst_end']
   log['late'][k]=late;log['synthetic_delay_ms'][k]=q['late_ms'] if late else q['on_time_ms']
   if late:u=uaccepted.copy();applied_fallback=accepted_fallback
   else:uaccepted=candidate.copy();accepted_fallback=applied_fallback
   if name=='MPC':ctl.prev=u-np.array([cc['plant']['m']*cc['plant']['g'],0,0,0])
  cmd,sat,rate=allocate(u,alloc,cmd,dt,slew);ctl.sat=sat or rate
  log['requested'][k]=u;log['candidate'][k]=candidate;log['commands'][k]=cmd
  fr=Wa@u;fm=alloc['kf']*alloc['omega_max']**2
  log['rotor_requested'][k]=fr;log['rotor_allocated'][k]=alloc['kf']*cmd**2;log['speed_requested'][k]=np.sqrt(np.maximum(fr,0)/alloc['kf'])
  log['sat'][k]=sat;log['rate'][k]=rate;log['solver_bad'][k]=bool(ctl.diag.get('fallback',False));log['applied_fallback'][k]=applied_fallback;log['solver_iter'][k]=ctl.iterations
  log['solver_status'][k]=ctl.status;log['motor_violation'][k]=np.any((om< -1e-8)|(om>p['omega_max']+1e-8))
  for key,val in ctl.diag.items():
   if key in log:log[key][k]=val
  log['smc_surface'][k]=ctl.smc_surface;log['smc_switch'][k]=ctl.smc_switch
  x,om=integrate(x,om,cmd,p,force[k],dt,c['substeps'])
  if name=='MPC':
   log['prediction_position_error'][k]=np.linalg.norm(x[:3]-ctl.linear_next[:3]);log['prediction_velocity_error'][k]=np.linalg.norm(x[3:6]-ctl.linear_next[3:6])
  err=np.linalg.norm(x[:3]-ref(t+dt,tr,c)[0])
  if not np.all(np.isfinite(x)):fail_reason='nonfinite_state'
  elif err>c['failure']['position_error_max']:fail_reason='position_error_above_10m'
  elif max(abs(x[6:8]))>c['failure']['attitude_max']:fail_reason='roll_or_pitch_above_1.2rad'
  if fail_reason:failed=True;fail_time=t+dt;fail_state=x.copy();break
  log['state_contract']=np.array(c['delay']['state_policy'])
 log.update(failed=np.array(failed),case=np.array(case),trajectory=np.array(tr),failure_time=np.array(fail_time),failure_reason=np.array(fail_reason),failure_state=fail_state)
 return log,p
def augmented_metrics(log,c):
 m=metrics(log,c);valid=np.isfinite(log['x'][:,0]);t=log['t'][valid];fb=log['solver_bad'][valid];e=np.linalg.norm(log['x'][valid,:3]-log['ref'][valid],axis=1)
 run=longest=0
 for b in fb:run=run+1 if b else 0;longest=max(longest,run)
 pre=t<c['scenarios']['step_time'];post=~pre
 m.update(fallback_ratio=float(np.mean(fb)),fallback_longest_s=longest*c['dt'],fallback_first_s=float(t[fb][0]) if fb.any() else np.nan,failure_time=float(log['failure_time']),failure_reason=str(log['failure_reason']),completion_fraction=float(valid.sum()/len(valid)),prestep_max_error=float(np.max(e[pre])) if pre.any() else np.nan,poststep_rmse=float(np.sqrt(np.mean(e[post]**2))) if post.any() else np.nan,synthetic_deadline_misses=int(np.sum(log['late'][valid])))
 u0=np.array([c['plant']['m']*c['plant']['g'],0,0,0]);L=np.array([u0[0]]+[u0[0]*c['plant']['l']]*3);requested=log['requested'][valid]/L
 m.update(requested_variation=float(np.sum(np.diff(requested,axis=0)**2)/c['dt']),applied_fallback_ratio=float(np.mean(log['applied_fallback'][valid])),max_abs_tilt=float(np.max(abs(log['x'][valid,6:8]))),runtime_p95_ms=float(np.percentile(log['time_ms'][valid],95)),runtime_p99_ms=float(np.percentile(log['time_ms'][valid],99)),metric_window='valid_partial_[0,failure_time)' if bool(log['failed']) else 'completed_[0,20)',poststep_window='valid_samples_t>=3')
 return m
def run_one(job):
 path=ROOT/'outputs/raw'/(job['id']+'.npz')
 if path.exists():
  with np.load(__import__('io').BytesIO(path.read_bytes())) as z:log={key:z[key] for key in z.files}
  metadata=json.loads(path.with_suffix('.json').read_text())
  if metadata['job']!=job:raise ValueError('Saved episode configuration differs; preserve the archive and use a fresh output directory: '+job['id'])
  p=metadata['actual_plant']
 else:
  log,p=simulate(job)
  np.savez_compressed(path,**log)
  path.with_suffix('.json').write_text(json.dumps(dict(job=job,actual_plant=p),indent=2))
 return log,p

def prepare_one(job):
 run_one(job)
 return job['id']

def main():
 c=json.loads((ROOT/'config.json').read_text());plan=jobs(c);out=ROOT/'outputs';raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
 (out/'execution_plan.json').write_text(json.dumps(plan,indent=2))
 import concurrent.futures
 pending=[j for j in plan if not (raw/(j['id']+'.npz')).exists()]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for i,key in enumerate(pool.map(prepare_one,pending)):
   if i%4==0:print('Prepared',i+1,'/',len(pending),key,flush=True)
 rows=[];manifest=[]
 for i,job in enumerate(plan):
  path=raw/(job['id']+'.npz')
  log,p=run_one(job)
  m=augmented_metrics(log,job['config']);m.update(id=job['id'],group=job['group'],controller=job['controller'],label=LABELS[job['controller']],trajectory=job['trajectory'],case=job['case'],variant=job['variant'],sample=job['sample']);rows.append(m)
  manifest.append(dict(controller=job['controller'],trajectory=job['trajectory'],case=job['case'],seed=job['config']['seed']+job['sample'],completion=not bool(log['failed']),failure_time=m['failure_time'],failure_reason=m['failure_reason'],metric_path='outputs/summary.csv',metric_row_id=job['id'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/n for n in ['simulation.py','controllers.py','plant.py','references.py','metrics.py']]},id=job['id'],raw=str(path.relative_to(ROOT)),config_sha256=hashlib.sha256(json.dumps(job['config'],sort_keys=True).encode()).hexdigest(),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  if i%4==0:print(i+1,'/',len(plan),job['id'],'RMSE',round(m['rmse_3d'],4),'F',m['failure'],flush=True)
 with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 env=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,osqp=osqp.__version__,platform=platform.platform(),processor=platform.processor(),thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},config_sha256=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest())
 try:
  env['cpu_model']=__import__('subprocess').check_output(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],text=True).strip()
 except Exception as ex:env['cpu_model']='unavailable: '+str(ex)
 (out/'environment.json').write_text(json.dumps(env,indent=2));print('DONE',len(rows),flush=True)
if __name__=='__main__':main()



