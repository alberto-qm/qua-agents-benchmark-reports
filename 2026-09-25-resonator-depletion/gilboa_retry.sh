#!/bin/bash
# Probe gilboa every 5 min (up to 18 times, 1.5 h); once a tiny job runs, run chain.sh then chain2.sh on it.
cd ~/qab-runs/depletion-20260925 || exit 1
PY=~/code/QM/tinycal/.venv/bin/python
for i in $(seq 1 18); do
  if $PY gilboa_probe.py > logs/gilboa_probe_$i.out 2>&1; then
    echo "$(date +%H:%M:%S) gilboa answered on probe $i; starting the chain" >> logs/gilboa_retry.log
    rm -f kappa/gilboa/r1_meta.json
    ./chain.sh gilboa > logs/chain_gilboa_retry.out 2>&1
    ./chain2.sh gilboa > logs/chain2_gilboa_retry.out 2>&1
    echo "$(date +%H:%M:%S) gilboa retry done" >> logs/gilboa_retry.log
    exit 0
  fi
  echo "$(date +%H:%M:%S) probe $i: $(grep -h 'gilboa down' logs/gilboa_probe_$i.out | cut -c1-160)" >> logs/gilboa_retry.log
  sleep 300
done
echo "$(date +%H:%M:%S) gilboa never answered; giving up" >> logs/gilboa_retry.log
