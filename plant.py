import numpy as np

def rot(eta):
 ph,th,ps=eta;c=np.cos;s=np.sin
 return np.array([[c(ps)*c(th),c(ps)*s(th)*s(ph)-s(ps)*c(ph),c(ps)*s(th)*c(ph)+s(ps)*s(ph)],[s(ps)*c(th),s(ps)*s(th)*s(ph)+c(ps)*c(ph),s(ps)*s(th)*c(ph)-c(ps)*s(ph)],[-s(th),c(th)*s(ph),c(th)*c(ph)]])

def emap(eta):
 ph,th,_=eta
 return np.array([[1,np.sin(ph)*np.tan(th),np.cos(ph)*np.tan(th)],[0,np.cos(ph),-np.sin(ph)],[0,np.sin(ph)/np.cos(th),np.cos(ph)/np.cos(th)]])

def mix(p):
 l=p['l'];c=p['km']/p['kf']
 return np.array([[1,1,1,1],[0,-l,0,l],[-l,0,l,0],[-c,c,-c,c]])

def rigid(x,u,p,f=None,mot=None):
 if f is None:f=np.zeros(3)
 J=np.array(p['J']);w=x[9:];gyro=np.zeros(3) if mot is None else p.get('I_M',0.)*np.cross(w,[0,0,1])*float(mot[0]-mot[1]+mot[2]-mot[3])
 return np.r_[x[3:6],rot(x[6:9])[:,2]*u[0]/p['m']-np.array([0,0,p['g']])+(f-np.array(p['drag'])*x[3:6])/p['m'],emap(x[6:9])@w,(u[1:]-np.cross(w,J*w)-gyro)/J]

def linear(p):
 A=np.zeros((12,12));B=np.zeros((12,4));A[:3,3:6]=np.eye(3);A[3:6,3:6]=-np.diag(p['drag'])/p['m'];A[3,7]=p['g'];A[4,6]=-p['g'];A[6:9,9:]=np.eye(3);B[5,0]=1/p['m'];B[9:,1:]=np.diag(1/np.array(p['J']))
 return A,B

def integrate(x,mot,cmd,p,f,dt,sub):
 M=mix(p);y=np.r_[x,mot];h=dt/sub
 def rhs(z):return np.r_[rigid(z[:12],M@(p['kf']*z[12:]**2),p,f,z[12:]),(cmd-z[12:])/p['tau_m']]
 for _ in range(sub):
  a=rhs(y);b=rhs(y+h*a/2);c=rhs(y+h*b/2);d=rhs(y+h*c);y+=h*(a+2*b+2*c+d)/6
 return y[:12],np.clip(y[12:],0,p['omega_max'])

def allocate(u,p,prev,dt,slew=np.inf):
 f=np.linalg.solve(mix(p),u);fm=p['kf']*p['omega_max']**2;unbounded=np.sqrt(np.clip(f,0,fm)/p['kf']);speed=np.clip(unbounded,prev-slew*dt,prev+slew*dt)
 return speed,bool(np.any((f<0)|(f>fm))),bool(np.any(abs(unbounded-prev)>slew*dt+1e-9))

