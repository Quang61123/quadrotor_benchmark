"""MPC variants (motor-state predictor, tilt-constraint policies, projected fallback) built on base_controllers.py."""
import numpy as np
import scipy.linalg as la
from scipy import sparse
import osqp
from base_controllers import Controller as Base
from plant import linear,mix
from references import flat

def prediction_model(p,dt,motor=False):
 A,B=linear(p)
 if motor:
  hover=np.sqrt(p['m']*p['g']/(4*p['kf']));G=mix(p)*(2*p['kf']*hover)
  Aa=np.zeros((16,16));Ba=np.zeros((16,4));Aa[:12,:12]=A;Aa[:12,12:]=B@G;Aa[12:,12:]=-np.eye(4)/p['tau_m'];Ba[12:]=np.linalg.inv(G)/p['tau_m'];A,B=Aa,Ba
 n=len(A);Z=np.zeros((n+4,n+4));Z[:n,:n]=A;Z[:n,n:]=B;E=la.expm(Z*dt)
 return E[:n,:n],E[:n,n:]

class Controller(Base):
 def setup(self):
  self.nx=12;self.backupK=self.K.copy();self.motor=self.s.get('predictor','rigid')=='motor';self.policy=self.s.get('tilt_policy','hard_initial');self.soft=self.policy=='soft_future';self.ns=2*self.s['N'] if self.soft else 0;self.hover=np.sqrt(self.p['m']*self.p['g']/(4*self.p['kf']))
  if self.motor:
   self.Ad,self.Bd=prediction_model(self.p,self.dt,True);self.nx=16;self.Q=la.block_diag(self.Q,np.zeros((4,4)));self.P=la.solve_discrete_are(self.Ad,self.Bd,self.Q,self.R)
  if not self.motor and self.policy=='hard_initial':super().setup()
  else:
   N=self.s['N'];nx=self.nx;self.nxv=(N+1)*nx;self.neq=self.nxv
   Qb=sparse.block_diag([self.Q]*N+[self.P]);self.Rb=sparse.kron(sparse.eye(N),self.R,format='csc');self.D=sparse.kron(sparse.diags([np.ones(N),-np.ones(N-1)],[0,-1],shape=(N,N)),sparse.eye(4),format='csc');self.Sb=sparse.kron(sparse.eye(N),np.diag(self.s['S']),format='csc')
   H=2*sparse.block_diag([Qb,self.Rb+self.D.T@self.Sb@self.D]+([sparse.eye(self.ns)*self.s.get('slack_penalty',10000.)] if self.soft else []),format='csc')
   Ax=sparse.kron(sparse.eye(N+1),-sparse.eye(nx))+sparse.kron(sparse.diags(np.ones(N),-1,shape=(N+1,N+1)),self.Ad);Bu=sparse.kron(sparse.vstack([sparse.csc_matrix((1,N)),sparse.eye(N)]),self.Bd);Ae=sparse.hstack([Ax,Bu,sparse.csc_matrix((self.neq,self.ns))])
   W=np.linalg.inv(mix(self.p));Ar=sparse.hstack([sparse.csc_matrix((4*N,self.nxv)),sparse.kron(sparse.eye(N),W),sparse.csc_matrix((4*N,self.ns))]);Ad=sparse.hstack([sparse.csc_matrix((4*N,self.nxv)),self.D,sparse.csc_matrix((4*N,self.ns))]);As=sparse.hstack([sparse.eye(self.nxv),sparse.csc_matrix((self.nxv,4*N+self.ns))])
   fh=W@np.array([self.p['m']*self.p['g'],0,0,0]);fm=self.p['kf']*self.p['omega_max']**2;sl=np.full(self.nxv,-np.inf);sh=np.full(self.nxv,np.inf)
   if not self.soft:
    for k in range(0 if self.policy=='hard_initial' else 1,N+1):sl[k*nx+6:k*nx+8]=-self.s['attitude_max'];sh[k*nx+6:k*nx+8]=self.s['attitude_max']
   self.lo=np.r_[np.zeros(self.neq),np.tile(-fh,N),-np.tile(self.s['du_max'],N),sl];self.hi=np.r_[np.zeros(self.neq),np.tile(fm-fh,N),np.tile(self.s['du_max'],N),sh];rows=[Ae,Ar,Ad,As]
   if self.soft:
    selector=sparse.coo_matrix((np.ones(self.ns),(np.arange(self.ns),[j*nx+a for j in range(1,N+1) for a in [6,7]])),shape=(self.ns,self.nxv)).tocsc()
    plus=sparse.hstack([selector,sparse.csc_matrix((self.ns,4*N)),-sparse.eye(self.ns)]);minus=sparse.hstack([-selector,sparse.csc_matrix((self.ns,4*N)),-sparse.eye(self.ns)]);nonneg=sparse.hstack([sparse.csc_matrix((self.ns,self.nxv+4*N)),sparse.eye(self.ns)]);rows += [plus,minus,nonneg]
    self.lo=np.r_[self.lo,np.full(2*self.ns,-np.inf),np.zeros(self.ns)];self.hi=np.r_[self.hi,np.full(2*self.ns,self.s['attitude_max']),np.full(self.ns,np.inf)]
   self.solver=osqp.OSQP();self.solver.setup(P=sparse.triu(H,format='csc'),q=np.zeros(H.shape[0]),A=sparse.vstack(rows,format='csc'),l=self.lo,u=self.hi,verbose=False,eps_abs=self.s['eps'],eps_rel=self.s['eps'],max_iter=self.s['max_iter'],polishing=False,warm_starting=True,adaptive_rho_interval=self.s['adaptive_rho_interval'])
  self.projector=None
  if self.s.get('backup_policy','raw')=='projected':
   scale=np.array([self.p['m']*self.p['g']]+[self.p['m']*self.p['g']*self.p['l']]*3);self.projection_weights=1/scale**2;W=np.linalg.inv(mix(self.p));self.projector=osqp.OSQP();self.projector.setup(P=sparse.diags(2*self.projection_weights,format='csc'),q=np.zeros(4),A=sparse.vstack([sparse.csc_matrix(W),sparse.eye(4)],format='csc'),l=np.zeros(8),u=np.ones(8),verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=4000,polishing=False,warm_starting=True,adaptive_rho_interval=50)
 def command(self,t,x):
  if self.name!='MPC':return super().command(t,x)
  self.qp_vector=None;self.qp_candidate=np.full(4,np.nan);self.backup_command=np.full(4,np.nan);self.smc_surface=np.full(3,np.nan);self.smc_switch=np.full(3,np.nan);self.predicted_next=np.full(12,np.nan);self.linear_next=np.full(12,np.nan)
  nx=self.nx;N=self.s['N'];uh=np.array([self.p['m']*self.p['g'],0,0,0]);state=np.r_[x,self.motor_state-self.hover] if self.motor else x
  refs=[flat(t+k*self.dt if self.s.get('preview',True) else t,self.tr,self.cfg) for k in range(N+1)];Xref=[np.r_[r[0],np.zeros(4)] if self.motor else r[0] for r in refs]
  qx=np.concatenate([-2*(self.P if k==N else self.Q)@Xref[k] for k in range(N+1)]);b=np.r_[self.prev,np.zeros(4*(N-1))];qu=-2*self.Rb@np.concatenate([r[1] for r in refs[:-1]])-2*self.D.T@self.Sb@b
  lo=self.lo.copy();hi=self.hi.copy();lo[:nx]=-state;hi[:nx]=-state;start=self.neq+4*N;lo[start:start+4]+=self.prev;hi[start:start+4]+=self.prev
  self.solver.update(q=np.r_[qx,qu,np.zeros(self.ns)],l=lo,u=hi);res=self.solver.solve();self.status=res.info.status;self.iterations=res.info.iter;ok=res.info.status_val in (1,2) and res.x is not None and np.all(np.isfinite(res.x));rotor_v=tilt_v=slackmax=initdef=dyndef=np.nan;inc_v=np.full(4,np.nan);physical=False
  if ok:
   self.qp_vector=res.x.copy();X=res.x[:self.nxv].reshape(N+1,nx);U=res.x[self.nxv:self.nxv+4*N].reshape(N,4);F=(U+uh)@np.linalg.inv(mix(self.p)).T;fm=self.p['kf']*self.p['omega_max']**2;rotor_v=float(np.maximum(np.maximum(-F,F-fm),0).max());inc_v=np.maximum(abs(np.diff(np.vstack([self.prev,U]),axis=0))-np.array(self.s['du_max']),0).max(axis=0)
   tilts=X[:,6:8] if self.policy=='hard_initial' else X[1:,6:8];slack=res.x[-self.ns:].reshape(N,2) if self.soft else np.zeros_like(tilts);slackmax=float(np.max(slack)) if self.soft else 0.;tilt_v=float(np.maximum(abs(tilts)-self.s['attitude_max']-slack,0).max());tol=self.s['physical_tolerances'];physical=rotor_v<=tol['rotor_N'] and tilt_v<=tol['tilt_rad'] and np.all(inc_v<=tol['increment']) and (not self.soft or np.min(slack)>=-tol['tilt_rad'])
   self.qp_candidate=U[0]+uh;self.predicted_next=X[1,:12].copy();initdef=float(np.max(abs(X[0]-state)));dyndef=float(np.max(abs(X[1:]-X[:-1]@self.Ad.T-U@self.Bd.T)))
  xr,uff=flat(t,self.tr,self.cfg);backup=uff-self.backupK@(x-xr)+uh;self.backup_command=backup.copy();fallback=not(ok and physical);u=backup if fallback else self.qp_candidate;project_bad=False;project_norm=0.
  if fallback and self.projector is not None:
   fm=self.p['kf']*self.p['omega_max']**2;du=np.array(self.s['du_max']);previous=self.prev+uh;self.projector.update(q=-2*self.projection_weights*backup,l=np.r_[np.zeros(4),previous-du],u=np.r_[np.full(4,fm),previous+du]);pr=self.projector.solve();project_bad=pr.info.status_val not in (1,2) or pr.x is None or not np.all(np.isfinite(pr.x));u=previous.copy() if project_bad else pr.x.copy();project_norm=float(np.linalg.norm(u-backup))
  reason=res.info.status if not ok else 'physical_tolerance_exceeded' if not physical else '';self.diag=dict(status_val=res.info.status_val,primal_residual=res.info.prim_res,dual_residual=res.info.dual_res,solve_time_ms=1000*res.info.solve_time,objective=res.info.obj_val,warm_start_enabled=True,warm_start_previous=t>0,fallback=fallback,fallback_reason=reason,current_attitude_outside_qp=bool(np.any(abs(x[6:8])>self.s['attitude_max'])),qp_accepted=not fallback,qp_rotor_violation_N=rotor_v,qp_tilt_violation_rad=tilt_v,qp_slack_max=slackmax,qp_initial_defect=initdef,qp_dynamics_defect=dyndef,projection_bad=project_bad,projection_norm=project_norm)
  for j in range(4):self.diag['qp_increment_violation_'+str(j)]=float(inc_v[j])
  self.prev=U[0].copy() if not fallback else u-uh
  self.linear_next=(self.Ad@state+self.Bd@self.prev)[:12]
  return u
