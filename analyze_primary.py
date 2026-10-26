"""Independent raw-horizon auditing and paired episode-level statistics."""
import os,sys,json,io,csv,hashlib,math,itertools
from pathlib import Path
ROOT=Path(__file__).parent;sys.path.insert(0,os.environ['QUAD_BENCH_DEPS'])
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.linalg import expm
from scipy.stats import binomtest
OUT=ROOT/'outputs'

def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def matrices(p,h,motor):
 A=np.zeros((12,12));B=np.zeros((12,4));A[:3,3:6]=np.eye(3);A[3:6,3:6]=-np.diag(p['drag'])/p['m'];A[3,7]=p['g'];A[4,6]=-p['g'];A[6:9,9:]=np.eye(3);B[5,0]=1/p['m'];B[9:,1:]=np.diag(1/np.array(p['J']))
 l=p['l'];a=p['km']/p['kf'];M=np.array([[1,1,1,1],[0,-l,0,l],[-l,0,l,0],[-a,a,-a,a]])
 if motor:
  wh=np.sqrt(p['m']*p['g']/(4*p['kf']));G=M*(2*p['kf']*wh);AA=np.zeros((16,16));BB=np.zeros((16,4));AA[:12,:12]=A;AA[:12,12:]=B@G;AA[12:,12:]=-np.eye(4)/p['tau_m'];BB[12:]=np.linalg.inv(G)/p['tau_m'];A,B=AA,BB
 n=len(A);Z=np.zeros((n+4,n+4));Z[:n,:n]=A;Z[:n,n:]=B;E=expm(Z*h);return E[:n,:n],E[:n,n:],M

def audit(label):
 plan=json.loads((OUT/(label+'_plan.json')).read_text());rows=[];exceptions=[];successful=selected=0
 for j in plan:
  if j['controller']!='MPC':continue
  c=j['config'];p=c['plant'].copy()
  if j['case']=='C1a':p['omega_max']=c['scenarios']['C1_omega_max']
  s=c['MPC'];motor=s.get('predictor','rigid')=='motor';nx=16 if motor else 12;N=s['N'];nv=(N+1)*nx;policy=s.get('tilt_policy','hard_initial');soft=policy=='soft_future'
  path=OUT/'raw'/(j['id']+'.npz');z=np.load(io.BytesIO(path.read_bytes()));log={k:z[k] for k in z.files};z.close()
  mask=np.isin(log['status_val'],[1,2]);finite=np.all(np.isfinite(log['qp_vector']),axis=1);nonfinite=int(np.sum(mask&~finite));mask &= finite
  X=log['qp_vector'][mask,:nv].reshape(-1,N+1,nx);U=log['qp_vector'][mask,nv:nv+4*N].reshape(-1,N,4);count=len(X);acc=log['qp_accepted'][mask];successful+=count;selected+=int(acc.sum())
  row=dict(id=j['id'],config_sha256=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),successful=count,accepted=int(acc.sum()),nonfinite_success=nonfinite)
  if count:
   Ad,Bd,M=matrices(p,c['dt'],motor);state=log['x'][mask]
   if motor:state=np.c_[state,log['motors'][mask]-np.sqrt(p['m']*p['g']/(4*p['kf']))]
   init=abs(X[:,0]-state);dyn=abs(X[:,1:]-X[:,:-1]@Ad.T-U@Bd.T);reprop=np.empty_like(X);reprop[:,0]=state
   for k in range(N):reprop[:,k+1]=reprop[:,k]@Ad.T+U[:,k]@Bd.T
   blocks=[('position_m',slice(0,3),1e-4),('velocity_m_s',slice(3,6),1e-4),('Euler_rad',slice(6,9),1e-4),('body_rate_rad_s',slice(9,12),1e-4)]+([('motor_rad_s',slice(12,16),1e-2)] if motor else [])
   selected_bad=np.zeros(count,bool)
   for name,ix,tol in blocks:
    im=init[:,ix].max(axis=1);dm=dyn[:,:,ix].max(axis=(1,2));row['init_max_'+name]=float(im.max());row['dyn_max_'+name]=float(dm.max());row['dyn_P95_'+name]=float(np.percentile(dm,95));selected_bad |= (im>tol)|(dm>tol)
   uh=np.array([p['m']*p['g'],0,0,0]);F=(U+uh)@np.linalg.inv(M).T;fm=p['kf']*p['omega_max']**2;rv=np.maximum(np.maximum(-F,F-fm),0).max(axis=(1,2));dv=np.maximum(abs(np.diff(np.concatenate([log['previous_rate_input'][mask,None,:],U],axis=1),axis=1))-s['du_max'],0).max(axis=1)
   tilt=X[:,:,6:8] if policy=='hard_initial' else X[:,1:,6:8];rt=reprop[:,:,6:8] if policy=='hard_initial' else reprop[:,1:,6:8];slack=log['qp_vector'][mask,-2*N:].reshape(-1,N,2) if soft else np.zeros_like(tilt)
   tv=np.maximum(abs(tilt)-s['attitude_max']-slack,0).max(axis=(1,2));rtv=np.maximum(abs(rt)-s['attitude_max']-slack,0).max(axis=(1,2));tol=s['physical_tolerances'];gate=(rv<=tol['rotor_N'])&(tv<=tol['tilt_rad'])&np.all(dv<=tol['increment'],axis=1)
   if soft:gate &= slack.min(axis=(1,2))>=-tol['tilt_rad']
   row.update(gate_mismatch=int(np.sum(gate!=acc)),selected_dynamics_bad=int(np.sum(selected_bad&acc)),selected_reprop_tilt_bad=int(np.sum((rtv>tol['tilt_rad'])&acc)),max_selected_reprop_excess=float(rtv[acc].max()) if acc.any() else None,rotor_max_selected=float(rv[acc].max()) if acc.any() else None)
   if row['gate_mismatch'] or row['selected_dynamics_bad'] or row['selected_reprop_tilt_bad']:exceptions.append(row)
  if s.get('backup_policy')=='projected':
   sel=log['fallback'];sel &= np.isfinite(log['x'][:,0]);req=log['requested'][sel];F=req@np.linalg.inv(matrices(p,c['dt'],False)[2]).T;violation=max(float(np.maximum(-F,0).max()),float(np.maximum(F-p['kf']*p['omega_max']**2,0).max())) if len(F) else 0
   inc=np.maximum(abs(req-np.array([p['m']*p['g'],0,0,0])-log['previous_rate_input'][sel])-s['du_max'],0)
   row.update(projected_fallback_calls=int(sel.sum()),projection_failure=int(log['projection_bad'][sel].sum()),projection_rotor_excess_N=violation,projection_increment_excess=float(inc.max()) if len(inc) else 0)
   for ii,component_name in enumerate(['collective_N','roll_Nm','pitch_Nm','yaw_Nm']):row['projection_increment_excess_'+component_name]=float(inc[:,ii].max()) if len(inc) else 0
  rows.append(row)
  if len(rows)%8==0:print('AUDIT',label,len(rows),flush=True)
 keys=list(dict.fromkeys(k for r in rows for k in r));writecsv(OUT/(label+'_horizon_audit.csv'),[{k:r.get(k,'') for k in keys} for r in rows])
 result=dict(label=label,episodes=len(plan),MPC_episodes=len(rows),successful=successful,accepted=selected,exceptions=exceptions,gate_mismatches=sum(r.get('gate_mismatch',0) for r in rows),selected_dynamics_bad=sum(r.get('selected_dynamics_bad',0) for r in rows),selected_reprop_tilt_bad=sum(r.get('selected_reprop_tilt_bad',0) for r in rows),tolerances='Retrospective audit: each rigid-state block 1e-4 native units; motor-speed block 1e-2 rad/s (~1e-4 N hover thrust); gate tolerances from frozen per-job config. Soft future gate uses logged slack; not hard-tilt safety.')
 (OUT/(label+'_horizon_audit.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='exceptions'}),flush=True);return result

def ci(values,rng,reps=10000):
 a=np.array(values,float)
 if len(a)==0:return None,None,None
 boot=a[rng.integers(0,len(a),size=(reps,len(a)))].mean(axis=1);return float(a.mean()),float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))

def stats():
 data=json.loads((OUT/'heldout_metrics.json').read_text());rng=np.random.default_rng(920001);summary=[];pairs=[]
 for track,name in itertools.product(['fixed','tuned'],['PID','LQR','SMC','MPC']):
  rr=sorted([r for r in data if r['track']==track and r['controller']==name],key=lambda r:r['sample']);good=[r for r in rr if not r['failure']];nf=sum(r['failure'] for r in rr);fc=binomtest(nf,len(rr)).proportion_ci(confidence_level=.95,method='exact');rm=ci([r['rmse_3d'] for r in good],rng);vv=ci([r['variation'] for r in good],rng)
  summary.append(dict(track=track,controller=name,n=len(rr),completed=len(good),failures=nf,failure_pct=100*nf/len(rr),failure_CI_low_pct=100*fc.low,failure_CI_high_pct=100*fc.high,mean_RMSE_complete=rm[0],RMSE_CI_low=rm[1],RMSE_CI_high=rm[2],mean_variation_complete=vv[0],variation_CI_low=vv[1],variation_CI_high=vv[2]))
 for name in ['PID','LQR','SMC','MPC']:
  a={r['sample']:r for r in data if r['track']=='fixed' and r['controller']==name};b={r['sample']:r for r in data if r['track']=='tuned' and r['controller']==name};common=[i for i in a if not a[i]['failure'] and not b[i]['failure']];rm=ci([b[i]['rmse_3d']-a[i]['rmse_3d'] for i in common],rng);vv=ci([b[i]['variation']-a[i]['variation'] for i in common],rng);fd=ci([b[i]['failure']-a[i]['failure'] for i in a],rng)
  pairs.append(dict(contrast='tuned_minus_fixed',controller=name,n_all=len(a),n_both_complete=len(common),RMSE_difference=rm[0],RMSE_CI_low=rm[1],RMSE_CI_high=rm[2],variation_difference=vv[0],variation_CI_low=vv[1],variation_CI_high=vv[2],failure_risk_difference=fd[0],risk_CI_low=fd[1],risk_CI_high=fd[2]))
 for name in ['LQR','SMC','MPC']:
  a={r['sample']:r for r in data if r['track']=='tuned' and r['controller']=='PID'};b={r['sample']:r for r in data if r['track']=='tuned' and r['controller']==name};common=[i for i in a if not a[i]['failure'] and not b[i]['failure']];rm=ci([b[i]['rmse_3d']-a[i]['rmse_3d'] for i in common],rng);vv=ci([b[i]['variation']-a[i]['variation'] for i in common],rng);fd=ci([b[i]['failure']-a[i]['failure'] for i in a],rng)
  pairs.append(dict(contrast='tuned_minus_tuned_PID',controller=name,n_all=len(a),n_both_complete=len(common),RMSE_difference=rm[0],RMSE_CI_low=rm[1],RMSE_CI_high=rm[2],variation_difference=vv[0],variation_CI_low=vv[1],variation_CI_high=vv[2],failure_risk_difference=fd[0],risk_CI_low=fd[1],risk_CI_high=fd[2]))
 writecsv(OUT/'heldout_statistics.csv',summary);writecsv(OUT/'paired_statistics.csv',pairs);(OUT/'statistics.json').write_text(json.dumps(dict(summary=summary,pairs=pairs,seed=920001,replicates=10000,scope='Episode/paired-realization resampling; marginal/paired complete-case error is conditional on completion; descriptive unadjusted intervals, no p-values or universal ranking'),indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':
 if '--mechanism-only' in sys.argv:audit('mechanism')
 else:
  for label in ['training','validation','mechanism','heldout']:audit(label)
  stats()

