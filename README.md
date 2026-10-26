# Quadrotor controller benchmark

Simulation benchmark for quadrotor trajectory-tracking controllers on a common nonlinear plant (16 states: rigid body plus four first-order rotor-speed states, rotor gyroscopic coupling, saturation). All controllers run at 50 Hz; the plant is integrated with RK4 at 5 ms substeps.

Controllers: cascade PID, hover LQR, boundary-layer sliding-mode control (SMC), linear MPC with LQR fallback (MPC, OSQP, horizon 25), MPC with reference preview (MPCP), and a geometric SO(3) tracking controller (GEO).

## Requirements

Python 3.12 with the pinned packages in `requirements.txt` (numpy 2.5.3, scipy 1.18.1, osqp 1.1.3, matplotlib 3.11.2). Set the environment variable `QUAD_BENCH_DEPS` to the folder that holds these packages (the analysis and verification scripts require it; the others default to `../.deps`). With `QUAD_BENCH_USE_ENV=1`, `simulation.py` uses the active environment instead. Numerical libraries are forced to one thread.

## Files

| File | Purpose |
|---|---|
| `config.json` | Plant parameters, controller defaults, scenarios, failure limits |
| `plant.py` | Rigid-body and rotor dynamics, mixing matrix, RK4 integration, allocation |
| `references.py` | Reference trajectories (hover, step, smooth step, circle, figure-eight), flat feedforward, disturbances |
| `metrics.py` | Tracking, effort, saturation, timing and step-response metrics |
| `base_controllers.py` | PID, hover LQR, boundary-layer SMC and rigid-predictor MPC |
| `controllers.py` | MPC variants: motor-state predictor, tilt-constraint policies, projected fallback |
| `geometric_controller.py` | Geometric SO(3) tracking controller |
| `simulation.py` | Closed-loop episode simulation and episode metrics; `python simulation.py` runs a scenario sweep |
| `run_primary.py` | Primary benchmark: 12-candidate tuning, MPC mechanism factorial (48 cells), 400 paired held-out episodes |
| `latency_replay.py` | Dedicated timing profiles and exogenous command-arrival replay (20 ms deadline, discard-and-hold) |
| `analyze_primary.py` | Horizon audit of MPC solutions and bootstrap / Clopper-Pearson statistics |
| `audit_mechanisms.py` | Read-only diagnostics of fallback switching and solver timing in stored logs |
| `independent_full.py` | Reruns every primary episode and replay and compares all non-timing arrays (tolerance 1e-12) |
| `verify_physics.py` | Unit checks of parameters, mixing, hover balance, gyroscopic term, motor Jacobian, zero-delay replay |
| `verify_release.py` | Consistency checks of stored outputs, selection and aggregate statistics |
| `run_confirmation.py` | Confirmation benchmark: fresh realizations, MPCP, GEO, nested budgets 12/24/48 |
| `analyze_confirmation.py` | Statistics and budget figure for the confirmation benchmark |
| `reproduction_check.py` | Re-executes four held-out episodes and compares them with stored metrics |
| `pi_timing.py`, `setup_pi.sh`, `run_pi.sh` | Timing profiles, replay and iteration-cap variants on an embedded Linux board (Raspberry Pi 5) |

## Running

```
python verify_physics.py
python run_primary.py                 # writes outputs/ (protocol, tuning, mechanism, held-out)
python latency_replay.py              # after run_primary.py; one process, no other load
python analyze_primary.py
python audit_mechanisms.py
python independent_full.py outputs
python verify_release.py
python run_confirmation.py --workers 5   # writes outputs/confirmation/
python analyze_confirmation.py
python reproduction_check.py
```

`run_primary.py` and `run_confirmation.py` reuse stored raw episodes only when the stored job configuration matches exactly; otherwise they stop. Use a fresh `outputs` folder for a new run.

On a Raspberry Pi 5 (64-bit Raspberry Pi OS): `./setup_pi.sh` once, then `./run_pi.sh --quick` (smoke test) and `./run_pi.sh` (three repeats, iteration caps 4000/400/100). The script pins one core, sets the performance governor and records temperature and throttling flags.

## Notes

Seeds are fixed in the code and `config.json`; non-timing results are deterministic for a given platform and package build. Timing fields (`time_ms`, `solve_time_ms`) depend on the machine and are excluded from reproduction checks. Floating-point results can differ slightly between x86-64 and ARM builds.
