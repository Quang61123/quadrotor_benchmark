"""Consistency checks of stored outputs, source hashes, tuning selection and aggregate statistics."""
import os,sys,json,hashlib,io,csv
from pathlib import Path
sys.path.insert(0,os.environ['QUAD_BENCH_DEPS'])
import numpy as np
R=Path(__file__).parent; O=R/'outputs'
def read(n):return json.loads((O/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 protocol=read('protocol_frozen.json');assert all(sha(R/n)==h for n,h in protocol['design_source_sha256'].items())
 assert sha(O/'protocol_frozen.json')==(O/'protocol_frozen.sha256').read_text().split()[0]
 selection=read('selection_locked.json'); failures=[]; max_r=max_v=0.; count=0
 for group,expected in [('training',96),('validation',12),('mechanism',48),('heldout',400)]:
  plan=read(group+'_plan.json');assert len(plan)==expected
  for job in plan:
   c=job['config'];z=np.load(io.BytesIO((O/'raw'/(job['id']+'.npz')).read_bytes()));m=read('raw/'+job['id']+'_metrics.json');e=z['x'][:,:3]-z['ref'];valid=np.isfinite(e).all(axis=1)
   rms=float(np.sqrt(np.mean(np.sum(e[valid]**2,axis=1))));p=c['plant'];scale=np.array([p['m']*p['g']]+[p['m']*p['g']*p['l']]*3);du=np.diff(z['u'][valid]/scale,axis=0);v=float(np.sum(du*du)/c['dt'])
   max_r=max(max_r,abs(rms-m['rmse_3d']));max_v=max(max_v,abs(v-m['variation']));assert abs(rms-m['rmse_3d'])<1e-12 and abs(v-m['variation'])<1e-8
   assert int(z['failed'])==m['failure'];count+=1
   if m['failure']:failures.append(dict(group=group,id=job['id'],controller=job['controller'],termination_s=m['failure_time'],partial_rmse_m=rms,ranking='excluded from complete-case ranking'))
   z.close()
 train=read('training_metrics.json');val=read('validation_metrics.json')
 for name in ['PID','LQR','SMC','MPC']:
  scores={i:np.mean([r['objective'] for r in train if r['controller']==name and r['candidate_index']==i]) for i in range(12)}
  top=sorted(scores,key=lambda i:(scores[i],i))[:3];vv=[r for r in val if r['controller']==name];assert sorted(r['candidate_index'] for r in vv)==sorted(top)
  assert min(vv,key=lambda r:(r['objective'],r['candidate_index']))['candidate_index']==selection[name]['candidate_index']
 plan=read('heldout_plan.json')
 for sample in range(50):
  jj=[j for j in plan if j['sample']==sample];assert len(jj)==8
  for key in ['plant','scenarios','seed']:assert all(j['config'][key]==jj[0]['config'][key] for j in jj)
 stats=read('statistics.json');met=read('heldout_metrics.json')
 for s in stats['summary']:
  rr=[r for r in met if r['track']==s['track'] and r['controller']==s['controller']];good=[r for r in rr if not r['failure']];assert len(good)==s['completed'] and sum(r['failure'] for r in rr)==s['failures']
  assert abs(np.mean([r['rmse_3d'] for r in good])-s['mean_RMSE_complete'])<1e-12
 for s in stats['pairs']:
  if s['contrast']!='tuned_minus_fixed':continue
  rr={ (r['track'],r['sample']):r for r in met if r['controller']==s['controller']};dd=[rr['tuned',i]['rmse_3d']-rr['fixed',i]['rmse_3d'] for i in range(50) if not rr['tuned',i]['failure'] and not rr['fixed',i]['failure']];assert len(dd)==s['n_both_complete'] and abs(np.mean(dd)-s['RMSE_difference'])<1e-12
 audits=[read(g+'_horizon_audit.json') for g in ['training','validation','mechanism','heldout']];assert all(a['gate_mismatches']==a['selected_dynamics_bad']==a['selected_reprop_tilt_bad']==0 for a in audits)
 indep=read('independent_full_report.json');assert indep['all_passed'] and indep['episodes']==562 and indep['max_numeric_difference']==0
 with (O/'failure_inventory.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(failures[0]));w.writeheader();w.writerows(failures)
 out=dict(all_passed=True,primary_episodes=count,primary_failures=len(failures),max_RMSE_recalculation_error=max_r,max_variation_recalculation_error=max_v,frozen_execution_sources_unchanged=True,selection_recomputed_without_heldout=True,paired_scenario_configs_checked=50,successful_candidates=sum(a['successful'] for a in audits),accepted_candidates=sum(a['accepted'] for a in audits),independent_episodes=562,independent_max_difference=0,scope='Retrospective consistency checks of stored outputs; the protocol hash is local, not externally registered.')
 (O/'release_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
