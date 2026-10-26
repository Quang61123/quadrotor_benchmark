import numpy as np

def ref(t,tr,cfg):
 s=cfg['scenarios'];w=s['circle_omega'];R=s['circle_radius'];zero=np.zeros(3)
 if tr=='novel8':
  A=s['figure8_A'];B=s['figure8_B'];C=s['figure8_C'];w=s['figure8_omega'];wz=s['figure8_vertical_omega'];ph=s.get('path_phase',0.)
  return np.array([A*np.sin(w*t+ph),B*np.sin(2*w*t+.7*ph),1+C*np.sin(wz*t+.3*ph)]),np.array([A*w*np.cos(w*t+ph),2*B*w*np.cos(2*w*t+.7*ph),C*wz*np.cos(wz*t+.3*ph)]),np.array([-A*w*w*np.sin(w*t+ph),-4*B*w*w*np.sin(2*w*t+.7*ph),-C*wz*wz*np.sin(wz*t+.3*ph)])
 if tr=='hover':return np.array([0.,0.,1.]),zero,zero
 if tr=='step':return np.array(s['step_target'] if t>=s['step_time'] else [0,0,1.],float),zero,zero
 if tr=='smooth':
  tau=np.clip((t-s['step_time'])/s['smooth_duration'],0.,1.);D=np.array(s['step_target'])-[0,0,1.]
  q=10*tau**3-15*tau**4+6*tau**5
  qd=(30*tau**2-60*tau**3+30*tau**4)/s['smooth_duration']
  qdd=(60*tau-180*tau**2+120*tau**3)/s['smooth_duration']**2
  return np.array([0.,0.,1.])+D*q,D*qd,D*qdd
 if tr=='circle':return np.array([R*np.cos(w*t),R*np.sin(w*t),1.]),np.array([-R*w*np.sin(w*t),R*w*np.cos(w*t),0.]),np.array([-R*w*w*np.cos(w*t),-R*w*w*np.sin(w*t),0.])
 A,B,C=s['figure8_A'],s['figure8_B'],s['figure8_C'];w=s['figure8_omega'];wz=s['figure8_vertical_omega']
 return np.array([A*np.sin(w*t),B*np.sin(2*w*t),1+C*np.sin(wz*t)]),np.array([A*w*np.cos(w*t),2*B*w*np.cos(2*w*t),C*wz*np.cos(wz*t)]),np.array([-A*w*w*np.sin(w*t),-4*B*w*w*np.sin(2*w*t),-C*wz*wz*np.sin(wz*t)])

def flat(t,tr,cfg):
 p,v,a=ref(t,tr,cfg);par=cfg['plant'];af=a+np.array(par['drag'])*v/par['m']
 return np.r_[p,v,-af[1]/par['g'],af[0]/par['g'],0.,np.zeros(3)],np.array([par['m']*af[2],0,0,0.])

def disturbance(ts,case,cfg,seed):
 s=cfg['scenarios'];f=np.zeros((len(ts),3))
 if case=='MC' and cfg.get('new_split'):
  rng=np.random.default_rng(seed+991);a,b=s['wind_interval'];f[(ts>=a)&(ts<b)]=np.array(s['wind_force'])*rng.uniform(.8,1.2);alpha=np.exp(-cfg['dt']/.25);noise=np.zeros(3)
  for k in range(len(ts)):
   noise=alpha*noise+np.sqrt(1-alpha**2)*rng.normal(size=3)*[.2,.2,.1];f[k]+=noise
 elif case in ('D1','MC'):a,b=s['wind_interval'];f[(ts>=a)&(ts<b)]=s['wind_force']
 elif case=='D2':
  a,b=s['gust_interval'];ix=(ts>=a)&(ts<b);f[ix]=np.sin(np.pi*(ts[ix]-a)/(b-a))[:,None]*np.array(s['gust_force'])
 elif case=='D3':
  rng=np.random.default_rng(seed);alpha=np.exp(-cfg['dt']/s['stochastic_tau'])
  for k in range(1,len(ts)):f[k]=alpha*f[k-1]+s['stochastic_sigma']*np.sqrt(1-alpha**2)*rng.normal(size=3)
  f[:,2]=0
 return f

