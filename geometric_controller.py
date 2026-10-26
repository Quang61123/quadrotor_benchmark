"""Geometric tracking controller on SO(3) (Lee, Leok and McClamroch, 2010) on the shared controller interface.
All other controller names are delegated unchanged to controllers.Controller."""
import numpy as np
from controllers import Controller as BaseController
from plant import rot
from references import ref

def vee(S):return np.array([S[2,1],S[0,2],S[1,0]])

class Controller(BaseController):
 def command(self,t,x):
  if self.name!='GEO':return super().command(t,x)
  self.status='ok';self.iterations=0;self.diag={};self.qp_vector=None;self.qp_candidate=np.full(4,np.nan);self.backup_command=np.full(4,np.nan);self.smc_surface=np.full(3,np.nan);self.smc_switch=np.full(3,np.nan)
  s=self.s;p=self.p;pr,vr,ar=ref(t,self.tr,self.cfg);e=x[:3]-pr;ev=x[3:6]-vr
  acc=ar-np.array(s['kx'])*e-np.array(s['kv'])*ev+np.array(p['drag'])*x[3:6]/p['m']
  F=p['m']*(acc+np.array([0,0,p['g']]))
  nF=np.linalg.norm(F);b3=F/nF if nF>1e-6 else np.array([0,0,1.])
  tilt=np.arccos(np.clip(b3[2],-1,1));tmax=s.get('tilt_max',.55)
  if tilt>tmax:
   h=b3[:2]/max(np.linalg.norm(b3[:2]),1e-9);b3=np.r_[np.sin(tmax)*h,np.cos(tmax)]
  b1c=np.array([1.,0,0]);b2=np.cross(b3,b1c);b2/=np.linalg.norm(b2);b1=np.cross(b2,b3);Rd=np.c_[b1,b2,b3]
  R=rot(x[6:9]);W=x[9:];eR=.5*vee(Rd.T@R-R.T@Rd);eW=W
  J=np.array(p['J']);tau=J*(-np.array(s['kR'])*eR-np.array(s['kW'])*eW)+np.cross(W,J*W)
  return np.r_[max(0.,F@R[:,2]),tau]
