#!/usr/bin/env bash
# Run the timing measurement. Usage: ./run_pi.sh            (full, ~20-40 min)
#                              ./run_pi.sh --quick    (smoke test, ~1 min)
set -e
cd "$(dirname "$0")"
source "$HOME/quadpi-venv/bin/activate"
# Fixed CPU clock: no frequency scaling during the measurement
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee "$g" >/dev/null; done
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)  $(vcgencmd measure_temp)  $(vcgencmd get_throttled)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 QUAD_BENCH_USE_ENV=1 PYTHONUNBUFFERED=1
# Pin to one core (core 3) with high priority, like the single-thread desktop run
sudo -E nice -n -10 taskset -c 3 "$HOME/quadpi-venv/bin/python" pi_timing.py "$@" 2>&1 | tee results_pi_log.txt
sudo chown -R "$USER" results_pi results_pi_log.txt 2>/dev/null || true
echo "Results: results_pi/ and results_pi_log.txt"
