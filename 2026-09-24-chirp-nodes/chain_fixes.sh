#!/bin/bash
# Start the fix tests on a backend once its chirp driver has finished (one job at a time per backend).
cd ~/qab-runs/chirp-nodes-20260924
PY=~/code/QM/tinycal/.venv/bin/python
b=$1
until grep -q "all steps done: r0,wide3a,wide3b,sat,fine,end" $b/driver.log 2>/dev/null; do sleep 15; done
if [ "$b" = gilboa ]; then
  $PY run_fixes.py gilboa fix ab20,ab80 > fixes_gilboa.out 2>&1
elif [ "$b" = qolab ]; then
  $PY run_fixes.py qolab main map03b,r09a > fixes_qolab_main.out 2>&1
  $PY run_fixes.py qolab fix map03b,r09a > fixes_qolab_fix.out 2>&1
fi
echo "chain $b finished" >> chain.log
