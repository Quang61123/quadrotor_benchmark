"""Quiet sequential profiles and exogenous command-arrival latency replay."""
import os,sys,json,copy,io,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).parent;sys.path.insert(0,os.environ['QUAD_BENCH_DEPS'])
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import simulation as runner
from controllers import Controller
from plant import integrate
from run_primary import BASE,job
OUT=ROOT/'outputs';D=OUT/'latency';D.mkdir(exist_ok=True)
IDS=['A_0_rigid_raw_nominal_step','A_1_rigid_raw_nominal_step','A_0_motor_raw_nominal_step','A_1_motor_raw_nominal_step','A_1_rigid_raw_C1a_step','A_1_motor_raw_C1a_step']

def replay(j,latency):
 c=j['config'];h=c['dt'];n=len(latency);state=dict(k=0,delay=0.,held=np.array([c['plant']['m']*c['plant']['g'],0,0,0]),held_fb=False,previous_cmd=np.full(4,np.sqrt(c['plant']['m']*c['plant']['g']/(4*c['plant']['kf']))));records=[]
 class Replayed(Controller):
  def command(self,t,x):
   candidate=super().command(t,x);k=state['k'];ms=float(latency[k]);late=ms>h*1000;fb=bool(self.diag.get('fallback',False))
   if late:
    selected=state['held'].copy();applied_fb=state['held_fb'];self.prev=selected-np.array([self.p['m']*self.p['g'],0,0,0])
   else:selected=candidate;applied_fb=fb;state['held']=selected.copy();state['held_fb']=fb
   state['delay']=h if late else ms/1000
   records.append(dict(start_s=float(t),finish_s=float(t+ms/1000),arrival_s=None if late else float(t+ms/1000),latency_ms=ms,late=late,candidate=candidate.tolist(),applied_fallback=applied_fb));state['k']+=1;return selected
 def split_integrate(x,mot,cmd,p,f,dt,sub):
  delay=state['delay'];old=state['previous_cmd'].copy();step=dt/sub
  if delay>0:x,mot=integrate(x,mot,old,p,f,delay,max(1,math.ceil(delay/step)))
  if dt-delay>1e-15:x,mot=integrate(x,mot,cmd,p,f,dt-delay,max(1,math.ceil((dt-delay)/step)))
  state['previous_cmd']=cmd.copy();return x,mot
 ctl_before=runner.Controller;int_before=runner.integrate
 try:runner.Controller=Replayed;runner.integrate=split_integrate;log,p=runner.simulate(j)
 finally:runner.Controller=ctl_before;runner.integrate=int_before
 valid=len(records);log['late'][:valid]=[r['late'] for r in records];log['candidate'][:valid]=[r['candidate'] for r in records];log['applied_fallback'][:valid]=[r['applied_fallback'] for r in records];log['synthetic_delay_ms'][:valid]=[r['latency_ms'] for r in records]
 return log,p,records

def main():
 assert (OUT/'heldout_metrics.json').exists(),'Run timing profiles only after run_primary.py has finished (no concurrent workers)'
 plan={j['id']:j for j in json.loads((OUT/'mechanism_plan.json').read_text())};rows=[];manifest=[]
 for key in IDS:
  j=copy.deepcopy(plan[key]);profpath=D/(key+'_profile.npz')
  if profpath.exists():z=np.load(io.BytesIO(profpath.read_bytes()));profile={k:z[k] for k in z.files};z.close()
  else:profile,p=runner.simulate(j);np.savez_compressed(profpath,**profile)
  baseline=np.load(io.BytesIO((OUT/'raw'/(key+'.npz')).read_bytes()))
  for item in ['x','u','requested','motors','commands','qp_vector','status_val','solver_bad']:np.testing.assert_allclose(profile[item],baseline[item],rtol=0,atol=1e-12,equal_nan=True)
  baseline.close();valid=np.isfinite(profile['x'][:,0]);timing=profile['time_ms'][valid];calls=len(timing);duration=calls*j['config']['dt'];pj=copy.deepcopy(j);pj['config']['duration']=duration;pj['group']='latency_replay';pj['id']='replay_'+key;pj['config']['latency_source']='outputs/latency/'+profpath.name
  replaypath=D/(key+'_replay.npz');tracepath=D/(key+'_arrival_trace.json')
  if replaypath.exists():z=np.load(io.BytesIO(replaypath.read_bytes()));delayed={k:z[k] for k in z.files};z.close();records=json.loads(tracepath.read_text())
  else:delayed,p,records=replay(pj,timing);np.savez_compressed(replaypath,**delayed);tracepath.write_text(json.dumps(records,indent=2))
  no=runner.augmented_metrics(profile,j['config']);de=runner.augmented_metrics(delayed,pj['config'])
  q=np.percentile(timing,[50,95,99]);rows.append(dict(id=key,preview=j['preview'],predictor=j['predictor'],case=j['case'],calls=calls,horizon_s=duration,p50_ms=float(q[0]),p95_ms=float(q[1]),p99_ms=float(q[2]),max_ms=float(timing.max()),measured_misses=int(np.sum(timing>20)),measured_miss_pct=100*float(np.mean(timing>20)),no_delay_failure=no['failure'],no_delay_post_rmse=no['poststep_rmse'],replay_failure=de['failure'],replay_failure_time=de['failure_time'],replay_calls=len(records),replay_misses=int(sum(r['late'] for r in records)),replay_post_rmse=de['poststep_rmse'],replay_applied_variation=de['variation']))
  manifest.append(dict(job=pj,source_profile=str(profpath.relative_to(ROOT)),profile_sha256=hashlib.sha256(profpath.read_bytes()).hexdigest(),replay=str(replaypath.relative_to(ROOT)),replay_sha256=hashlib.sha256(replaypath.read_bytes()).hexdigest(),trace=str(tracepath.relative_to(ROOT)),state_trace_matches_original=True))
  print('PROFILE/REPLAY',key,'P99',q[2],'misses',np.sum(timing>20),'replay F',de['failure'],flush=True)
 # Zero-arrival integration reproduces the baseline on a short circle fixture.
 c=copy.deepcopy(BASE);c['duration']=1.;j=job('zero_fixture',c,'MPC','circle','nominal','verification');gold,p=runner.simulate(j);zero,p,trace=replay(j,np.zeros(50))
 for key in ['x','u','requested','motors','commands','qp_vector','status_val']:np.testing.assert_allclose(zero[key],gold[key],rtol=0,atol=1e-12,equal_nan=True)
 (D/'zero_arrival_verification.json').write_text(json.dumps(dict(passed=True,tolerance=1e-12,calls=50)))
 (OUT/'latency_metrics.json').write_text(json.dumps(rows,indent=2));(OUT/'latency_manifest.json').write_text(json.dumps(dict(profiles=manifest,scope='Dedicated sequential closed-loop command timing includes reference generation and numerical diagnostics; excludes setup, allocation, integration and logging. One numerical thread; no warm-up; first call included; no concurrent study workers. Exogenous timing trace is replayed with within-sample actuator command arrival; late candidates discarded and last command held. Controller computation/state commit is atomic offline; no CPU queue/co-simulation or hardware/SIL certificate. Finite failed-profile horizon reused for both outcomes; not a 20s completion claim.',source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['controllers.py','base_controllers.py','simulation.py','latency_replay.py','plant.py','references.py']},python=sys.version,platform=__import__('platform').platform()),indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
