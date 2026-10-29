# Quadrotor Controller Benchmark

This repository contains the source code and reproducibility package for the manuscript:

**How Tuning, Reference Preview, Actuator Modeling, and Fallback Policies Change Quadrotor Controller Comparisons: A Reproducible Simulation Study**

**Xuan-Quy Dao, Le-Minh Nguyen, and Ngoc-Bich Le**

**arXiv:2610.35165 [cs.RO]**

## Paper

The manuscript presents a reproducible comparison of quadrotor trajectory-tracking controllers under a common nonlinear simulation framework.

The evaluated controllers include:

- Cascade PID
- Hover LQR
- Boundary-layer sliding-mode control (BL-SMC)
- Linear MPC with LQR fallback (LMPC-F)
- Preview-enabled LMPC-F
- Geometric SO(3) tracking control

Rather than comparing controllers using tracking error alone, the study examines how controller rankings are affected by:

- tuning effort and tuning budget,
- reference preview,
- actuator-state prediction,
- optimizer fallback policies,
- actuator constraints,
- command timing and computational deadlines.

The benchmark uses a common nonlinear quadrotor plant including rigid-body dynamics, four first-order rotor-speed states, rotor gyroscopic coupling, motor lag, and actuator saturation.

All controllers operate at **50 Hz**, while the nonlinear plant is integrated using RK4 with **5 ms substeps**.

## Citation

If you use this repository, benchmark, implementation, experimental protocol, or results in your research, please cite the associated manuscript:

```bibtex
@article{dao2026quadrotorbenchmark,
  title   = {How Tuning, Reference Preview, Actuator Modeling, and Fallback Policies Change Quadrotor Controller Comparisons: A Reproducible Simulation Study},
  author  = {Dao, Xuan-Quy and Nguyen, Le-Minh and Le, Ngoc-Bich},
  journal = {arXiv preprint arXiv:2610.35165},
  year    = {2026},
  eprint  = {2610.35165},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi     = {10.48550/arXiv.2610.35165}
}
```

For publications using a particular controller implementation, simulation configuration, tuning protocol, or evaluation dataset from this repository, please cite the manuscript above and reference this repository in the reproducibility statement.

## Reproducibility Scope

The repository supports the experiments reported in the manuscript, including:

1. matched controller tuning,
2. paired held-out trajectory evaluation,
3. MPC mechanism factorial experiments,
4. reference-preview experiments,
5. actuator-predictor and fallback-policy analysis,
6. confirmation experiments using fresh realizations,
7. nested tuning-budget comparisons,
8. command-arrival latency replay,
9. Raspberry Pi 5 timing experiments, and
10. independent numerical and physics verification.

The primary study includes matched 12-candidate tuning, paired held-out evaluations, and a 48-cell MPC mechanism factorial.

The confirmation study evaluates fresh realizations using nested tuning budgets of 12, 24, and 48 candidates and additionally includes the preview-enabled MPC and geometric SO(3) controllers.

## Requirements

The reference environment uses **Python 3.12** with the package versions pinned in `requirements.txt`.

Main dependencies include:

```text
numpy
scipy
osqp
matplotlib
```

For strict reproduction, use the versions specified in `requirements.txt`.

The environment variable `QUAD_BENCH_DEPS` may be set to the directory containing the required packages.

For example:

```bash
export QUAD_BENCH_DEPS=/path/to/dependencies
```

Alternatively,

```bash
export QUAD_BENCH_USE_ENV=1
```

allows `simulation.py` to use the active Python environment.

Numerical libraries are restricted to a single thread where required to improve reproducibility.

## Repository Structure

| File | Purpose |
|---|---|
| `config.json` | Plant parameters, controller defaults, scenarios, and failure limits |
| `plant.py` | Quadrotor rigid-body dynamics, rotor dynamics, mixing, allocation, and RK4 integration |
| `references.py` | Hover, step, smooth-step, circle, and figure-eight references; disturbances and feedforward terms |
| `metrics.py` | Tracking, control-effort, saturation, timing, and step-response metrics |
| `base_controllers.py` | Cascade PID, hover LQR, BL-SMC, and rigid-predictor MPC |
| `controllers.py` | MPC variants, motor-state predictor, tilt policies, and projected fallback |
| `geometric_controller.py` | Geometric SO(3) trajectory-tracking controller |
| `simulation.py` | Closed-loop simulation and episode-level metric computation |
| `run_primary.py` | Primary tuning, mechanism-factorial, and paired evaluation experiments |
| `analyze_primary.py` | Primary-study statistical analysis and MPC solution audit |
| `audit_mechanisms.py` | Diagnostics for fallback switching and solver timing |
| `latency_replay.py` | Exogenous command-arrival replay and deadline experiments |
| `independent_full.py` | Independent rerun and numerical comparison of primary experiments |
| `verify_physics.py` | Physics and implementation consistency checks |
| `verify_release.py` | Stored-output, selection, and aggregate-statistic verification |
| `run_confirmation.py` | Fresh-realization confirmation study and nested tuning-budget experiments |
| `analyze_confirmation.py` | Confirmation-study statistics and tuning-budget analysis |
| `reproduction_check.py` | Lightweight reproduction test using held-out episodes |
| `pi_timing.py` | Raspberry Pi timing experiments |
| `setup_pi.sh` | Raspberry Pi environment setup |
| `run_pi.sh` | Raspberry Pi timing and replay experiments |

## Running the Benchmark

### 1. Verify the physical model

```bash
python verify_physics.py
```

### 2. Run the primary study

```bash
python run_primary.py
```

This generates the primary experiment outputs, including tuning results, MPC mechanism experiments, and paired held-out evaluations.

### 3. Run the latency replay

```bash
python latency_replay.py
```

For meaningful timing measurements, run this process without unrelated computational load.

### 4. Analyze the primary study

```bash
python analyze_primary.py
python audit_mechanisms.py
```

### 5. Perform independent verification

```bash
python independent_full.py outputs
python verify_release.py
```

### 6. Run the confirmation study

```bash
python run_confirmation.py --workers 5
```

The confirmation outputs are written under:

```text
outputs/confirmation/
```

### 7. Analyze the confirmation experiment

```bash
python analyze_confirmation.py
```

### 8. Run the lightweight reproduction check

```bash
python reproduction_check.py
```

## Raspberry Pi 5 Timing Experiments

For a 64-bit Raspberry Pi OS installation:

```bash
./setup_pi.sh
```

Run a short smoke test:

```bash
./run_pi.sh --quick
```

Then run the full timing experiment:

```bash
./run_pi.sh
```

The timing protocol evaluates multiple solver iteration limits and records processor temperature and throttling status.

Because execution time depends strongly on hardware, operating system, CPU scheduling, and background processes, timing results should not be expected to reproduce bit-for-bit across platforms.

## Reproducibility Notes

Random seeds are fixed in the source code and configuration files.

For a fixed software environment and compatible numerical platform, the non-timing quantities are intended to be deterministic.

Timing-related quantities such as

```text
time_ms
solve_time_ms
```

are hardware dependent and are therefore excluded from strict numerical reproduction checks.

Small floating-point differences can occur between architectures, particularly between x86-64 and ARM systems.

`run_primary.py` and `run_confirmation.py` reuse stored raw episodes only when the stored configuration exactly matches the requested experiment.

For a new experiment or modified configuration, use a fresh `outputs/` directory.

## Recommended Reproduction Sequence

For verification of the results reported in the manuscript, the recommended sequence is:

```bash
python verify_physics.py
python run_primary.py
python latency_replay.py
python analyze_primary.py
python audit_mechanisms.py
python independent_full.py outputs
python verify_release.py
python run_confirmation.py --workers 5
python analyze_confirmation.py
python reproduction_check.py
```

## Research Use

This repository is intended to support transparent and reproducible research on quadrotor controller benchmarking.

When extending the benchmark, we recommend reporting at least:

- tracking accuracy,
- tuning protocol and tuning budget,
- completion and failure rates,
- fallback activity,
- actuator saturation,
- applied-input variation,
- solver timing,
- command timing/deadline violations, and
- computational platform.

These quantities are important because controller rankings can change when tuning effort, reference information, fallback behavior, actuator modeling, or implementation timing changes.

## Manuscript

**X.-Q. Dao, L.-M. Nguyen, and N.-B. Le**,  
*How Tuning, Reference Preview, Actuator Modeling, and Fallback Policies Change Quadrotor Controller Comparisons: A Reproducible Simulation Study*,  
arXiv:2610.35165, 2026.

## Contact

For questions concerning the benchmark, implementation, or reproduction of the manuscript results, please open an issue in this repository.

## Acknowledgment

If this repository contributes to a publication, thesis, benchmark comparison, or derivative implementation, we would appreciate citation of the associated manuscript.