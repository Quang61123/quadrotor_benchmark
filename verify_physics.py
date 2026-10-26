"""Physics and interface checks: parameters, rotor mixing, hover balance, gyroscopic term, motor-hover Jacobian, zero-delay replay."""
import os,sys,json,io
from pathlib import Path
R=Path(__file__).parent;sys.path.insert(0,os.environ['QUAD_BENCH_DEPS'])
import numpy as np
from plant import mix,rigid,integrate,linear
from controllers import prediction_model
from run_primary import BASE,job
from simulation import simulate
from latency_replay import replay
p=BASE['plant'];expected=dict(m=.468,l=.225,J=[.004856,.004856,.008801],kf=2.98e-6,km=1.14e-7,tau_m=.03,omega_max=1000.,drag=[.25]*3,I_M=3.357e-5)
assert all(p[k]==v for k,v in expected.items())
M=np.array([[1,1,1,1],[0,-p['l'],0,p['l']],[-p['l'],0,p['l'],0],[-p['km']/p['kf'],p['km']/p['kf'],-p['km']/p['kf'],p['km']/p['kf']]])
np.testing.assert_array_equal(mix(p),M)
x=np.zeros(12);x[2]=1;wh=np.sqrt(p['m']*p['g']/(4*p['kf']));om=np.full(4,wh);u=M@(p['kf']*om**2);np.testing.assert_allclose(rigid(x,u,p,mot=om),0,atol=1e-14)
xx,mm=integrate(x.copy(),om.copy(),om,p,np.zeros(3),1.,200);np.testing.assert_allclose(xx,x,atol=1e-12);np.testing.assert_allclose(mm,om,atol=1e-12)
x[9:]=[.2,-.3,.1];om=np.array([600.,500.,550.,450.]);du=rigid(x,u,p,mot=om)[9:]-rigid(x,u,p,mot=np.full(4,wh))[9:]
expected_gyro=-p['I_M']*np.array([-.3,-.2,0])*(600-500+550-450)/p['J'];np.testing.assert_allclose(du,expected_gyro,atol=1e-12)
x=np.zeros(12);G=M*(2*p['kf']*wh);eps=.001
for i in range(4):
 op=np.full(4,wh);on=op.copy();op[i]+=eps;on[i]-=eps
 der=(rigid(x,M@(p['kf']*op**2),p,mot=op)-rigid(x,M@(p['kf']*on**2),p,mot=on))/(2*eps);np.testing.assert_allclose(der,linear(p)[1]@G[:,i],atol=1e-8,rtol=1e-8)
import copy
c=copy.deepcopy(BASE);c['duration']=1.;j=job('zero_arrival_fixture',c,'MPC','circle','nominal','verification');a,_=simulate(j);b,_,_=replay(j,np.zeros(50))
for k in ['x','u','requested','motors','commands','qp_vector','status_val']:np.testing.assert_allclose(a[k],b[k],atol=1e-12,rtol=0,equal_nan=True)
result=dict(all_passed=True,physical_config_matches_expected=True,rotor_matrix_matches_expected=True,hover_balance=True,gyro_direct_equation=True,motor_hover_finite_difference=True,zero_arrival_50_calls_match=True,scope='Implementation tests of the plant model and replay interface; not aircraft identification.')
(R/'outputs/implementation_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
