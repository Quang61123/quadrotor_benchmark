import numpy as np

def metrics(log,cfg):
 dt=cfg['dt'];e=log['x'][:,:3]-log['ref'];nn=np.linalg.norm(e,axis=1);good=np.isfinite(nn);rms=np.sqrt(np.mean(e[good]**2,axis=0));times=log['time_ms'][good];p=cfg['plant'];scale=np.array([p['m']*p['g'],p['m']*p['g']*p['l'],p['m']*p['g']*p['l'],p['m']*p['g']*p['l']]);un=log['u'][good]/scale
 r=dict(rmse_x=rms[0],rmse_y=rms[1],rmse_z=rms[2],rmse_3d=np.sqrt(np.mean(nn[good]**2)),iae=np.sum(nn[good])*dt,ise=np.sum(nn[good]**2)*dt,max_error=np.max(nn[good]),effort=np.sum(un**2)*dt,variation=np.sum((np.diff(un,axis=0)/dt)**2)*dt,saturation_ratio=np.mean(log['sat'][good]),saturation_duration=np.sum(log['sat'][good])*dt,rate_ratio=np.mean(log['rate'][good]),time_mean_ms=np.mean(times),time_p95_ms=np.percentile(times,95),time_max_ms=np.max(times),deadline_ratio=np.max(times)/(dt*1000),deadline_miss_ratio=np.mean(times>dt*1000),failure=int(log['failed']),solver_failures=int(np.sum(log['solver_bad'])),rotor_violations=int(np.sum(log['motor_violation'])),completed_duration=int(good.sum())*dt,recovery_time=np.nan,rise_time=np.nan,settling_time=np.nan,overshoot_pct=np.nan,steady_error=np.mean(nn[good][-int(2/dt):]))
 if str(log['case']) in ('D1','D2','MC') and not log['failed']:
  tend=cfg['scenarios']['gust_interval'][1] if str(log['case'])=='D2' else cfg['scenarios']['wind_interval'][1]
  for k in np.where(log['t']>=tend)[0]:
   if k+int(1/dt)<=len(nn) and np.all(nn[k:k+int(1/dt)]<=.1):r['recovery_time']=log['t'][k]-tend;break
 if str(log['trajectory'])=='step' and not log['failed']:
  ts=cfg['scenarios']['step_time'];idx=log['t']>=ts;d=np.array(cfg['scenarios']['step_target'])-[0,0,1];prog=(log['x'][idx,:3]-[0,0,1])@d/(d@d);tt=log['t'][idx]-ts;r['overshoot_pct']=max(0.,(max(prog)-1)*100);a=tt[prog>=.1];b=tt[prog>=.9]
  if len(a) and len(b):r['rise_time']=b[0]-a[0]
  ne=nn[idx]
  for k in range(len(ne)):
   if np.all(ne[k:]<=.02*np.linalg.norm(d)):r['settling_time']=tt[k];break
 return r

