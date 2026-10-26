"""Statistics and budget figure for the confirmation benchmark (numpy/scipy/matplotlib).
Usage: python analyze_confirmation.py [outputs/confirmation]"""
import json,sys,itertools,csv
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
D=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'outputs'/'confirmation'
fresh=json.loads((D/'fresh_metrics.json').read_text());sel=json.loads((D/'selection.json').read_text())
CTRL=['PID','LQR','SMC','MPC','GEO','MPCP'];B=[12,24,48];rng=np.random.default_rng(930001)
def ci(a,reps=10000):
 a=np.asarray(a,float)
 if len(a)==0:return [None]*3
 b=a[rng.integers(0,len(a),(reps,len(a)))].mean(1);return [float(a.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]
def rows(name,track):return {r['sample']:r for r in fresh if r['method']==name and r['track']==track}
def track_of(name,budget):
 if budget==0:return 'fixed'
 i=sel[f'{name}@{budget}']['candidate_index'];return 'fixed' if i==0 else f'cand{i:02d}'
summary=[];paired=[];budget=[]
for name in CTRL:
 for b in [0]+B:
  rr=rows(name,track_of(name,b));good=[r for r in rr.values() if not r['failure']];nf=sum(r['failure'] for r in rr.values());fc=binomtest(nf,len(rr)).proportion_ci(method='exact')
  e=dict(method=name,budget=b,candidate=0 if b==0 else sel[f'{name}@{b}']['candidate_index'],n=len(rr),failures=nf,fail_lo=100*fc.low,fail_hi=100*fc.high,rmse=ci([r['rmse_3d'] for r in good]),V=ci([r['variation'] for r in good]),fallback_pct=float(100*np.mean([r['fallback_ratio'] for r in good])) if name in('MPC','MPCP') and good else None)
  summary.append(e)
 f=rows(name,'fixed');t=rows(name,track_of(name,12));common=[i for i in f if i in t and not f[i]['failure'] and not t[i]['failure']]
 paired.append(dict(method=name,pairs=len(common),dRMSE=ci([t[i]['rmse_3d']-f[i]['rmse_3d'] for i in common]),dV=ci([t[i]['variation']-f[i]['variation'] for i in common])))
 for b in [24,48]:
  a=rows(name,track_of(name,12));c=rows(name,track_of(name,b));common=[i for i in a if i in c and not a[i]['failure'] and not c[i]['failure']]
  budget.append(dict(method=name,budget=b,same_selection=track_of(name,b)==track_of(name,12),pairs=len(common),dRMSE_vs12=ci([c[i]['rmse_3d']-a[i]['rmse_3d'] for i in common])))
# MPCP vs MPC at budget 12, GEO vs PID
def pair(n1,n2,b=12):
 a=rows(n1,track_of(n1,b));c=rows(n2,track_of(n2,b));common=[i for i in a if i in c and not a[i]['failure'] and not c[i]['failure']];return dict(contrast=f'{n2}-{n1}@{b}',pairs=len(common),dRMSE=ci([c[i]['rmse_3d']-a[i]['rmse_3d'] for i in common]),dV=ci([c[i]['variation']-a[i]['variation'] for i in common]))
contr=[pair('MPC','MPCP'),pair('PID','GEO'),pair('SMC','GEO'),pair('MPC','MPCP',48)]
# ranking stability
rank={b:[e['method'] for e in sorted([e for e in summary if e['budget']==b],key=lambda e:e['rmse'][0])] for b in [0]+B}
res=dict(summary=summary,paired_tuned12_minus_fixed=paired,budget_vs12=budget,contrasts=contr,ranking_by_budget=rank,seed=930001,replicates=10000)
(D/'statistics.json').write_text(json.dumps(res,indent=1));print(json.dumps(dict(rank=rank,contr=contr),indent=1))
for e in summary:print(e['method'],e['budget'],e['candidate'],e['failures'],[round(x,4) for x in e['rmse']],round(e['V'][0],3),e['fallback_pct'])
# figure: budget curve
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'serif'})
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
cols=dict(PID='#1f77b4',LQR='#d95f02',SMC='#1b9e77',MPC='#7570b3',GEO='#e7298a',MPCP='#66a61e')
lab=dict(PID='PID',LQR='LQR',SMC='BL-SMC',MPC='LMPC-F',GEO='Geometric',MPCP='LMPC-F+preview')
xs=[0,12,24,48];xp=np.arange(4)
for k,name in enumerate(CTRL):
 m=[e for e in summary if e['method']==name];y=[e['rmse'][0] for e in m];lo=[e['rmse'][0]-e['rmse'][1] for e in m];hi=[e['rmse'][2]-e['rmse'][0] for e in m]
 ax[0].errorbar(xp+(k-2.5)*0.05,y,yerr=[lo,hi],marker='o',ms=3,lw=1,capsize=2,color=cols[name],label=lab[name])
ax[0].set_yscale('log');ax[0].set_xticks(xp);ax[0].set_xticklabels(['fixed','12','24','48']);ax[0].set_xlabel('Tuning budget (candidates)');ax[0].set_ylabel('Mean RMSE on fresh set (m)');ax[0].grid(alpha=.3,which='both');ax[0].legend(fontsize=6,ncol=2,frameon=False)
for k,name in enumerate(CTRL):
 m=[e for e in summary if e['method']==name];ax[1].plot(xp+(k-2.5)*0.05,[e['V'][0] for e in m],marker='s',ms=3,lw=1,color=cols[name])
ax[1].set_yscale('log');ax[1].set_xticks(xp);ax[1].set_xticklabels(['fixed','12','24','48']);ax[1].set_xlabel('Tuning budget (candidates)');ax[1].set_ylabel('Mean applied-input variation $V$ (s$^{-1}$)');ax[1].grid(alpha=.3,which='both')
fig.tight_layout();fig.savefig(D/'budget_fresh.pdf');print('FIG')
