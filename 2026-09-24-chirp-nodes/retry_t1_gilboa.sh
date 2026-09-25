#!/bin/bash
# Retry t1_chirp_test.py on gilboa every 5 min (its QOP gateway refused connections at 15:29 and 15:32), up to 6 tries.
cd ~/qab-runs/chirp-nodes-20260924
for i in 1 2 3 4 5 6; do
  sleep 300
  ~/code/QM/tinycal/.venv/bin/python t1_chirp_test.py gilboa > t1_chirp_gilboa_retry$i.out 2>&1
  if [ -f t1_chirp/gilboa_qD2.npz ] && [ -f t1_chirp/gilboa_qC3.npz ] && [ -f t1_chirp/gilboa_qD5.npz ]; then
    echo "gilboa done on try $i"; exit 0
  fi
  echo "try $i failed: $(grep -o 'Gateway health[^,]*' t1_chirp_gilboa_retry$i.out | head -1)"
done
exit 1
