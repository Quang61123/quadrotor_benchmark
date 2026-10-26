"""Read-only diagnostics of archived controller switching and solver timing."""
from pathlib import Path
import os,sys,json,collections,hashlib
if os.environ.get('QUAD_BENCH_DEPS'):sys.path.insert(0,os.environ['QUAD_BENCH_DEPS'])
import numpy as np
R=Path(__file__).parent;O=R/'outputs'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
 rows=json.loads((O/'heldout_metrics.json').read_text());out={'data_scope':'Post hoc diagnostics of archived logs; no new simulation or retuning','heldout':{},'profiles':{}}
 for track in ['fixed','tuned']:
  selected=[r for r in rows if r['controller']=='MPC' and r['track']==track];assert len(selected)==50
  episodes=[];statuses=collections.Counter();reasons=collections.Counter();total=fbcount=outtilt=trans=0;vswitch=vtotal=vfb=0
  for row in selected:
   z=np.load(O/'raw'/f"{row['id']}.npz");good=np.all(np.isfinite(z['x']),axis=1);fb=z['fallback'][good];n=len(fb);t=z['t'][good];xx=z['x'][good];cfg=json.loads((O/'raw'/f"{row['id']}.json").read_text())['job']['config'];p=cfg['plant'];scale=np.array([p['m']*p['g']]+[p['m']*p['g']*p['l']]*3);energy=np.sum(np.diff(z['u'][good]/scale,axis=0)**2,axis=1)/cfg['dt'];switch=fb[1:]!=fb[:-1];nearfallback=fb[1:]|fb[:-1]
   assert abs(energy.sum()-row['variation'])<1e-8
   total+=n;fbcount+=int(fb.sum());outtilt+=int(np.any(abs(xx[:,6:8])>.45,axis=1).sum());trans+=int(switch.sum());vtotal+=float(energy.sum());vswitch+=float(energy[switch].sum());vfb+=float(energy[nearfallback].sum());statuses.update(map(int,z['status_val'][good]));reasons.update(map(str,z['fallback_reason'][good][fb]));episodes.append(dict(id=row['id'],fallback_pct=100*float(fb.mean()),switches=int(switch.sum()),variation=float(energy.sum()),switch_adjacent_variation=float(energy[switch].sum()),fallback_adjacent_variation=float(energy[nearfallback].sum())))
  out['heldout'][track]=dict(calls=total,fallback_calls=fbcount,mean_fallback_pct=100*fbcount/total,initial_tilt_outside_calls=outtilt,status_counts=dict(statuses),fallback_reasons=dict(reasons),switches=trans,mean_switches=trans/50,switch_adjacent_V_fraction=vswitch/vtotal,fallback_adjacent_V_fraction=vfb/vtotal,mean_V=vtotal/50,episodes=episodes)
 for p in sorted((O/'latency').glob('*nominal_step_profile.npz')):
  z=np.load(p);good=np.isfinite(z['time_ms']);it=z['solver_iter'][good];tm=z['time_ms'][good];solve=z['solve_time_ms'][good];sv=z['status_val'][good];tail=tm>20
  out['profiles'][p.stem]=dict(calls=len(it),command_p50_ms=float(np.median(tm)),command_p99_ms=float(np.percentile(tm,99)),solver_p50_ms=float(np.median(solve)),solver_p99_ms=float(np.percentile(solve,99)),iteration_p50=float(np.median(it)),iteration_p95=float(np.percentile(it,95)),iteration_p99=float(np.percentile(it,99)),iteration_max=int(it.max()),at_4000=int((it==4000).sum()),under_150_pct=float(np.mean(it<150)*100),status_counts=dict(collections.Counter(map(int,sv))),misses=int(tail.sum()),misses_at_4000=int(np.sum(tail&(it==4000))),iteration_time_corr=float(np.corrcoef(it,tm)[0,1]))
 protocol=json.loads((O/'protocol_frozen.json').read_text());out['frozen_sources']={n:digest(R/n) for n in protocol['design_source_sha256']};assert out['frozen_sources']==protocol['design_source_sha256']
 # Hash all primary raw outputs, timings, bootstrap and selected settings; no input is written.
 evidence=list((O/'raw').glob('*'))+list((O/'latency').glob('*'))+[O/n for n in ['statistics.json','selection_locked.json','protocol_frozen.json','heldout_metrics.json','latency_metrics.json']]
 out['evidence_sha256']={p.relative_to(R).as_posix():digest(p) for p in evidence if p.is_file()}
 (O/'mechanism_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 for key in ['heldout','profiles']:
  print(key,json.dumps({k:{a:b for a,b in v.items() if a!='episodes'} for k,v in out[key].items()}),flush=True)
if __name__=='__main__':audit()

