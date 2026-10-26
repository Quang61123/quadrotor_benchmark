"""Command-call timing and command-arrival replay on an embedded Linux board (tested on Raspberry Pi 5).

For each of the six MPC timing configurations it
  1. runs a dedicated closed-loop profile on this computer and records the
     command-call time of every 50 Hz step (reference generation and diagnostics
     included; setup, allocation, integration and logging excluded),
  2. replays the measured trace with the arrival rule of latency_replay.py
     (late > 20 ms: discard and hold; otherwise switch the motor command at the
     measured arrival time),
  3. optionally repeats this with OSQP iteration caps (deadline-aware variant).
Results are written to ./results_pi/.
"""
import os, sys, json, copy, time, platform, argparse, subprocess, hashlib
from pathlib import Path
ROOT = Path(__file__).parent
os.environ.setdefault('QUAD_BENCH_DEPS', str(ROOT))      # packages come from the venv
os.environ['QUAD_BENCH_USE_ENV'] = '1'
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import numpy as np, scipy, osqp
import simulation as runner
from run_primary import BASE, job
from latency_replay import replay

OUT = ROOT / 'results_pi'; OUT.mkdir(exist_ok=True)
CONFIGS = [  # (id, preview, predictor, case) as in the mechanism plan of run_primary.py
    ('A_0_rigid_raw_nominal_step', False, 'rigid', 'nominal'),
    ('A_1_rigid_raw_nominal_step', True,  'rigid', 'nominal'),
    ('A_0_motor_raw_nominal_step', False, 'motor', 'nominal'),
    ('A_1_motor_raw_nominal_step', True,  'motor', 'nominal'),
    ('A_1_rigid_raw_C1a_step',     True,  'rigid', 'C1a'),
    ('A_1_motor_raw_C1a_step',     True,  'motor', 'C1a'),
]
# Reference measurements on a Windows desktop (Intel Core i5-8400, one numerical thread)
DESKTOP = {
    'A_0_rigid_raw_nominal_step': dict(p99=101.8, miss=8.9, nodelay='complete', replay='F 3.56'),
    'A_1_rigid_raw_nominal_step': dict(p99=100.1, miss=6.9, nodelay='complete', replay='F 2.88'),
    'A_0_motor_raw_nominal_step': dict(p99=32.2,  miss=1.5, nodelay='complete', replay='complete'),
    'A_1_motor_raw_nominal_step': dict(p99=95.1,  miss=5.9, nodelay='complete', replay='F 3.10'),
    'A_1_rigid_raw_C1a_step':     dict(p99=43.8,  miss=7.1, nodelay='F 3.36',   replay='no F in window'),
    'A_1_motor_raw_C1a_step':     dict(p99=47.0,  miss=8.3, nodelay='F 4.36',   replay='F 4.12'),
}

def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception as e: return f'unavailable ({e})'

def system_info():
    return dict(platform=platform.platform(), machine=platform.machine(), python=sys.version,
                numpy=np.__version__, scipy=scipy.__version__, osqp=getattr(osqp, '__version__', '?'),
                model=sh('cat /proc/device-tree/model 2>/dev/null | tr -d "\\0"'),
                governor=sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
                freq_khz=sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'),
                temp=sh('vcgencmd measure_temp'), throttled=sh('vcgencmd get_throttled'),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

def make_job(key, preview, predictor, case, cap):
    c = copy.deepcopy(BASE)
    c['MPC'].update(preview=preview, predictor=predictor, backup_policy='raw', tilt_policy='hard_initial')
    if cap is not None: c['MPC']['max_iter'] = int(cap)
    return job(key, c, 'MPC', 'step', case, 'mechanism', preview=preview, predictor=predictor,
               backup_policy='raw', tilt_policy='hard_initial')

def outcome(m):
    return 'complete' if not m['failure'] else f"F {m['failure_time']:.2f}"

def run_one(key, preview, predictor, case, cap, rep):
    j = make_job(key, preview, predictor, case, cap)
    t0 = time.perf_counter(); profile, _ = runner.simulate(j); wall = time.perf_counter() - t0
    valid = np.isfinite(profile['x'][:, 0]); timing = profile['time_ms'][valid]; solver = profile['solve_time_ms'][valid]
    calls = len(timing); dt = j['config']['dt']
    pj = copy.deepcopy(j); pj['config']['duration'] = calls * dt; pj['group'] = 'latency_replay'; pj['id'] = 'replay_' + key
    delayed, _, records = replay(pj, timing)
    no = runner.augmented_metrics(profile, j['config']); de = runner.augmented_metrics(delayed, pj['config'])
    q = np.percentile(timing, [50, 95, 99]); it = profile['solver_iter'][valid]
    tag = f"{key}_cap{cap or 4000}_r{rep}"
    np.savez_compressed(OUT / f'{tag}_timing.npz', time_ms=timing, solve_time_ms=solver, solver_iter=it)
    return dict(id=key, case=case, preview=preview, predictor=predictor, max_iter=cap or 4000, repeat=rep,
                calls=calls, window_s=round(calls * dt, 2), p50_ms=float(q[0]), p95_ms=float(q[1]), p99_ms=float(q[2]),
                max_ms=float(timing.max()), solver_p99_ms=float(np.nanpercentile(solver, 99)),
                iter_p99=float(np.percentile(it, 99)), iter_cap_hits=int(np.sum(it >= (cap or 4000))),
                miss_pct=100 * float(np.mean(timing > 20)), fallback_pct=100 * float(np.mean(profile['solver_bad'][valid])),
                nodelay_outcome=outcome(no), nodelay_post_rmse=no['poststep_rmse'],
                replay_outcome=('no F in window' if (not de['failure'] and calls * dt < 19.99) else outcome(de)),
                replay_post_rmse=de['poststep_rmse'], replay_misses=int(sum(r['late'] for r in records)), replay_calls=len(records),
                profile_wall_s=wall, temp_after=sh('vcgencmd measure_temp'), throttled_after=sh('vcgencmd get_throttled'))

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--repeats', type=int, default=3, help='independent repetitions of every profile')
    a.add_argument('--caps', default='4000,400,100', help='OSQP iteration caps; 4000 = default solver limit in config.json')
    a.add_argument('--quick', action='store_true', help='one config, one repeat (smoke test)')
    g = a.parse_args(); caps = [int(x) for x in g.caps.split(',')]
    info = system_info(); print(json.dumps(info, indent=1), flush=True)
    if 'performance' not in info['governor']: print('WARNING: CPU governor is not "performance"; timings will be noisy.', flush=True)
    rows = []; configs = CONFIGS[:1] if g.quick else CONFIGS; reps = 1 if g.quick else g.repeats
    for cap in caps:
        for key, pr, pd, case in configs:
            if cap != 4000 and case != 'nominal': continue      # caps studied on the nominal step only
            for rep in range(reps):
                r = run_one(key, pr, pd, case, None if cap == 4000 else cap, rep); rows.append(r)
                d = DESKTOP[key]
                print(f"{key:28s} cap={r['max_iter']:4d} rep={rep}  P99={r['p99_ms']:6.1f} ms (desktop {d['p99']})  "
                      f"miss={r['miss_pct']:4.1f}%  no-delay={r['nodelay_outcome']} (desktop {d['nodelay']})  "
                      f"replay={r['replay_outcome']} (desktop {d['replay']})  {r['temp_after']}", flush=True)
                (OUT / 'pi_timing_results.json').write_text(json.dumps(dict(system=info, desktop_reference=DESKTOP, rows=rows), indent=1))
    info['end'] = dict(temp=sh('vcgencmd measure_temp'), throttled=sh('vcgencmd get_throttled'))
    src = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ['controllers.py', 'base_controllers.py', 'simulation.py', 'latency_replay.py', 'plant.py', 'references.py', 'pi_timing.py']}
    (OUT / 'pi_timing_results.json').write_text(json.dumps(dict(system=info, desktop_reference=DESKTOP, source_sha256=src, rows=rows), indent=1))
    import csv
    with open(OUT / 'pi_timing_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('DONE ->', OUT, flush=True)

if __name__ == '__main__':
    main()
