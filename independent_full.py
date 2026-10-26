"""Independent rerun of every primary-benchmark episode and arrival replay; compares all non-timing arrays with the stored outputs.
Usage: python independent_full.py <outputs directory>"""
import os,sys,json,io,hashlib,concurrent.futures,time
from pathlib import Path
ROOT=Path(__file__).parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['QUAD_BENCH_USE_ENV']='1'
import numpy as np
import simulation

def compare(payload):
 j,reference=payload
 if j['group']=='latency_replay':
  from latency_replay import replay
  key=j['id'].removeprefix('replay_');profile=np.load(io.BytesIO((Path(reference)/'latency'/(key+'_profile.npz')).read_bytes()));latency=profile['time_ms'][np.isfinite(profile['x'][:,0])];profile.close();log,p,records=replay(j,latency);path=Path(reference)/'latency'/(key+'_replay.npz')
 else:log,p=simulation.simulate(j);path=Path(reference)/'raw'/(j['id']+'.npz')
 z=np.load(io.BytesIO(path.read_bytes()));maxdiff=0.;count=0
 for k in z.files:
  if k in ['time_ms','solve_time_ms']:continue
  a,b=log[k],z[k]
  if a.dtype.kind in 'biufc':
   np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,equal_nan=True,err_msg=j['id']+' '+k)
   values=abs(a.astype(float)-b.astype(float));finite=values[np.isfinite(values)];maxdiff=max(maxdiff,float(finite.max()) if len(finite) else 0.) if values.ndim else max(maxdiff,float(values) if np.isfinite(values) else 0.)
  else:assert np.array_equal(a,b),(j['id'],k)
  count+=1
 z.close();return dict(id=j['id'],arrays_checked=count,max_numeric_difference=maxdiff,config_sha256=hashlib.sha256(json.dumps(j['config'],sort_keys=True).encode()).hexdigest(),reference_raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),passed=True)

def main():
 reference=Path(sys.argv[1]);plans=[]
 for label in ['training','validation','mechanism','heldout']:plans += json.loads((reference/(label+'_plan.json')).read_text())
 primary_count=len(plans)
 latency_manifest=json.loads((reference/'latency_manifest.json').read_text());plans += [m['job'] for m in latency_manifest['profiles']]
 rows=[];start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(compare,[(j,str(reference)) for j in plans]):
   rows.append(r)
   if len(rows)%16==0:print('INDEPENDENT',len(rows),'/',len(plans),flush=True)
 report=dict(primary_episodes=primary_count,latency_replay_episodes=len(plans)-primary_count,episodes=len(rows),all_passed=True,tolerance=1e-12,timing_fields_excluded=['time_ms','solve_time_ms'],max_numeric_difference=max(r['max_numeric_difference'] for r in rows),rows=rows,elapsed_s=time.perf_counter()-start,python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__,osqp=__import__('osqp').__version__,module_paths={name:str(__import__(name).__file__) for name in ['numpy','scipy','osqp','controllers','simulation']},source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['controllers.py','base_controllers.py','plant.py','references.py','metrics.py','simulation.py','run_primary.py','latency_replay.py','independent_full.py']},environment_scope='Rerun with pinned package versions; numerical arrays compared with tolerance 1e-12, timing fields excluded.')
 (reference/'independent_full_report.json').write_text(json.dumps(report,indent=2));print('INDEPENDENT COMPLETE',len(rows),report['max_numeric_difference'],flush=True)
if __name__=='__main__':main()
