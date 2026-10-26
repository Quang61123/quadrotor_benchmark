import numpy as np
import scipy.linalg as la
from scipy import sparse
import osqp
from plant import rot, linear, mix
from references import ref, flat

class Controller:
 def __init__(self,name,cfg,tr):
  self.name=name;self.cfg=cfg;self.p=cfg['plant'];self.dt=cfg['dt'];self.tr=tr;self.s=cfg[name];self.integ=np.zeros(3);self.vf=np.zeros(3);self.sat=False;self.prev=np.zeros(4);self.status='ok';self.iterations=0
  if name in ('LQR','MPC'):
   A,B=linear(self.p);Z=np.zeros((16,16));Z[:12,:12]=A;Z[:12,12:]=B;E=la.expm(Z*self.dt);self.Ad=E[:12,:12];self.Bd=E[:12,12:];self.Q=np.diag(self.s['Q']);self.R=np.diag(self.s['R']);self.P=la.solve_discrete_are(self.Ad,self.Bd,self.Q,self.R);self.K=np.linalg.solve(self.R+self.Bd.T@self.P@self.Bd,self.Bd.T@self.P@self.Ad)
  if name=='MPC':self.setup()
 def setup(self):
  N=self.s['N'];self.nxv=(N+1)*12;self.neq=self.nxv
  Qb=sparse.block_diag([self.Q]*N+[self.P]);self.Rb=sparse.kron(sparse.eye(N),self.R,format='csc');self.D=sparse.kron(sparse.diags([np.ones(N),-np.ones(N-1)],[0,-1],shape=(N,N)),sparse.eye(4),format='csc');self.Sb=sparse.kron(sparse.eye(N),np.diag(self.s['S']),format='csc')
  H=2*sparse.block_diag([Qb,self.Rb+self.D.T@self.Sb@self.D],format='csc')
  Ax=sparse.kron(sparse.eye(N+1),-sparse.eye(12))+sparse.kron(sparse.diags(np.ones(N),-1,shape=(N+1,N+1)),self.Ad);Bu=sparse.kron(sparse.vstack([sparse.csc_matrix((1,N)),sparse.eye(N)]),self.Bd);Ae=sparse.hstack([Ax,Bu])
  W=np.linalg.inv(mix(self.p));Ar=sparse.hstack([sparse.csc_matrix((4*N,self.nxv)),sparse.kron(sparse.eye(N),W)]);Ad=sparse.hstack([sparse.csc_matrix((4*N,self.nxv)),self.D]);Astate=sparse.hstack([sparse.eye(self.nxv),sparse.csc_matrix((self.nxv,4*N))]);A=sparse.vstack([Ae,Ar,Ad,Astate],format='csc')
  fh=W@np.array([self.p['m']*self.p['g'],0,0,0]);fm=self.p['kf']*self.p['omega_max']**2
  self.lo=np.r_[np.zeros(self.neq),np.tile(-fh,N),-np.tile(self.s['du_max'],N)];self.hi=np.r_[np.zeros(self.neq),np.tile(fm-fh,N),np.tile(self.s['du_max'],N)]
  sl=np.full(self.nxv,-np.inf);sh=np.full(self.nxv,np.inf)
  for k in range(N+1):sl[k*12+6:k*12+8]=-self.s['attitude_max'];sh[k*12+6:k*12+8]=self.s['attitude_max']
  self.lo=np.r_[self.lo,sl];self.hi=np.r_[self.hi,sh]
  self.solver=osqp.OSQP();self.solver.setup(P=sparse.triu(H,format='csc'),q=np.zeros(H.shape[0]),A=A,l=self.lo,u=self.hi,verbose=False,eps_abs=self.s['eps'],eps_rel=self.s['eps'],max_iter=self.s['max_iter'],polishing=False,warm_starting=True,adaptive_rho_interval=self.s['adaptive_rho_interval'])
 def command(self,t,x):
  self.status='ok';self.iterations=0;self.diag={};self.qp_vector=None;self.qp_candidate=np.full(4,np.nan);self.backup_command=np.full(4,np.nan);self.smc_surface=np.full(3,np.nan);self.smc_switch=np.full(3,np.nan)
  if self.name in ('PID','SMC'):
   pr,vr,ar=ref(t,self.tr,self.cfg);e=x[:3]-pr;ev=x[3:6]-vr;s=self.s
   if self.name=='PID':
    alpha=np.exp(-self.dt/s['filter_tau']);self.vf=alpha*self.vf+(1-alpha)*ev
    if not self.sat:self.integ=np.clip(self.integ+e*self.dt,-s['integral_limit'],s['integral_limit'])
    ac=ar-np.array(s['kp'])*e-np.array(s['kd'])*self.vf-np.array(s['ki'])*self.integ
   else:
    lam=np.array(s['lambda_pos']);surface=ev+lam*e;ac=ar-lam*ev-np.array(s['k_pos'])*np.clip(surface/np.array(s['boundary_pos']),-1,1)
   if self.name=='SMC':self.smc_surface=surface.copy();self.smc_switch=-np.array(s['k_pos'])*np.clip(surface/np.array(s['boundary_pos']),-1,1)
   ac+=np.array(self.p['drag'])*x[3:6]/self.p['m'];force=self.p['m']*(ac+np.array([0,0,self.p['g']]))
   th=np.arctan2(force[0],max(force[2],.5));ph=np.arctan2(-force[1],np.hypot(force[0],force[2]));eta=np.clip([ph,th,0],[-.55,-.55,-np.inf],[.55,.55,np.inf]);ea=(x[6:9]-eta+np.pi)%(2*np.pi)-np.pi
   if self.name=='PID':ang=-np.array(s['att_kp'])*ea-np.array(s['att_kd'])*x[9:]
   else:
    lam=np.array(s['lambda_att']);ang=-lam*x[9:]-np.array(s['k_att'])*np.clip((x[9:]+lam*ea)/np.array(s['boundary_att']),-1,1)
   J=np.array(self.p['J']);return np.r_[max(0.,force@rot(x[6:9])[:,2]),J*ang+np.cross(x[9:],J*x[9:])]
  xr,uff=flat(t,self.tr,self.cfg)
  if self.name=='LQR':du=uff-self.K@(x-xr)
  else:
   N=self.s['N'];refs=[flat(t+k*self.dt if self.s.get('preview',True) else t,self.tr,self.cfg) for k in range(N+1)];qx=np.concatenate([-2*(self.P if k==N else self.Q)@refs[k][0] for k in range(N+1)]);b=np.r_[self.prev,np.zeros(4*(N-1))];qu=-2*self.Rb@np.concatenate([r[1] for r in refs[:-1]])-2*self.D.T@self.Sb@b
   lo=self.lo.copy();hi=self.hi.copy();lo[:12]=-x;hi[:12]=-x;start=self.neq+4*N;lo[start:start+4]+=self.prev;hi[start:start+4]+=self.prev
   self.solver.update(q=np.r_[qx,qu],l=lo,u=hi);res=self.solver.solve();self.status=res.info.status;self.iterations=res.info.iter
   accepted_status=res.info.status_val in (1,2) and res.x is not None and np.all(np.isfinite(res.x))
   physical=False;rotor_v=tilt_v=np.nan;inc_v=np.full(4,np.nan)
   if accepted_status:
    self.qp_vector=res.x.copy();X=res.x[:self.nxv].reshape(N+1,12);U=res.x[self.nxv:].reshape(N,4)
    uh=np.array([self.p['m']*self.p['g'],0,0,0]);F=(U+uh)@np.linalg.inv(mix(self.p)).T;fm=self.p['kf']*self.p['omega_max']**2
    rotor_v=float(np.max(np.maximum(np.maximum(-F,F-fm),0)))
    delta=np.diff(np.vstack([self.prev,U]),axis=0);inc_v=np.maximum(abs(delta)-np.array(self.s['du_max']),0).max(axis=0)
    tilt_v=float(np.maximum(abs(X[:,6:8])-self.s['attitude_max'],0).max())
    tol=self.s['physical_tolerances'];physical=rotor_v<=tol['rotor_N'] and tilt_v<=tol['tilt_rad'] and np.all(inc_v<=tol['increment'])
    self.qp_candidate=U[0]+uh
   fallback=not (accepted_status and physical)
   self.backup_command=uff-self.K@(x-xr)+np.array([self.p['m']*self.p['g'],0,0,0])
   reason=res.info.status if not accepted_status else 'physical_tolerance_exceeded' if not physical else ''
   self.diag=dict(status_val=res.info.status_val,primal_residual=res.info.prim_res,dual_residual=res.info.dual_res,solve_time_ms=1000*res.info.solve_time,objective=res.info.obj_val,warm_start_enabled=True,warm_start_previous=t>0,fallback=fallback,fallback_reason=reason,current_attitude_outside_qp=bool(np.any(abs(x[6:8])>self.s['attitude_max'])),qp_accepted=not fallback,qp_rotor_violation_N=rotor_v,qp_tilt_violation_rad=tilt_v)
   for j in range(4):self.diag['qp_increment_violation_'+str(j)]=float(inc_v[j])
   du=self.backup_command-np.array([self.p['m']*self.p['g'],0,0,0]) if fallback else res.x[self.nxv:self.nxv+4]
   self.prev=du.copy()
  return du+np.array([self.p['m']*self.p['g'],0,0,0])


