#!/bin/bash
# After chain.sh <backend> finishes: the long-pulse steady-state sweep (kappa_long.py). usage: chain2.sh <backend>
be=$1
cd ~/qab-runs/depletion-20260925 || exit 1
until [ -f logs/chain_${be}.done ]; do sleep 20; done
~/code/QM/tinycal/.venv/bin/python kappa_long.py $be > logs/kappa_long_${be}.out 2>&1
echo "$(date +%H:%M:%S) chain2 done" >> logs/chain2_${be}.done
