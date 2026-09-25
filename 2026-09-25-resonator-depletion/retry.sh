#!/bin/bash
# Round-2 jobs lost to cloud errors (ReadTimeout / 503) at 18:27-18:28. usage: retry.sh
cd ~/qab-runs/depletion-20260925 || exit 1
PY=~/code/QM/tinycal/.venv/bin/python
$PY kappa.py qolab 2 --only=Q2,Q5 > logs/retry_kappa_qolab_r2.out 2>&1 &
until [ -f logs/chain2_arbel.done ]; do sleep 15; done
$PY run_02b.py arbel A 2 --only=qD1 > logs/retry_02b_arbel_A_r2.out 2>&1
$PY run_02a.py arbel 2 --only=qB4 > logs/retry_02a_arbel_r2.out 2>&1
wait
echo "$(date +%H:%M:%S) retries done" >> logs/retry.done
