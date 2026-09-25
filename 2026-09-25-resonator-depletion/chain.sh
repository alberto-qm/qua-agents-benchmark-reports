#!/bin/bash
# Depletion-wait and kappa tests on one backend (25 Sep 2026). usage: chain.sh <backend>
# Round 1: kappa, 02a, 02b A B C D; round 2 in reverse order (02b D C B A, 02a, kappa), so drift does not
# line up with the condition. Each step is its own process; a failed step does not stop the chain.
be=$1
cd ~/qab-runs/depletion-20260925 || exit 1
PY=~/code/QM/tinycal/.venv/bin/python
$PY kappa.py $be 1 > logs/kappa_${be}_r1.out 2>&1
$PY run_02a.py $be 1 > logs/02a_${be}_r1.out 2>&1
for c in A B C D; do $PY run_02b.py $be $c 1 > logs/02b_${be}_${c}_r1.out 2>&1; done
for c in D C B A; do $PY run_02b.py $be $c 2 > logs/02b_${be}_${c}_r2.out 2>&1; done
$PY run_02a.py $be 2 > logs/02a_${be}_r2.out 2>&1
$PY kappa.py $be 2 > logs/kappa_${be}_r2.out 2>&1
echo "$(date +%H:%M:%S) chain done" >> logs/chain_${be}.done
