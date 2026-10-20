#!/bin/bash
# upstream-merge gate: pause the v41 calibration after its current layer, run the p600 ppl gate with the merged
# build (ds4-upstream), then resume the calibration.  Baseline (pre-merge build): ppl 10.47-10.64, 16.8-17.3 tok/s.
set -u
O=$HOME/AI/Strata-DS4-up/bench/upstream-merge-2026-10-10
CTL=$HOME/AI/Strata-DS4/tools/v41/calibctl.sh
PIDF=/media/Crucial1TB/v41-calib/driver.pid
R=$HOME/AI/Strata-DS4
echo "$(date '+%F %T') gate: asking calibration to stop after its layer" | tee -a $O/gate.log
$CTL stop >> $O/gate.log 2>&1
while [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; do sleep 20; done
echo "$(date '+%F %T') gate: calibration paused; running p600 with the merged build" | tee -a $O/gate.log
cd $R && bash tools/ds4/memguard.sh 80 78 -- $HOME/AI/Strata-DS4-up/build-gpu/ds4_generate \
  -m /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf --ids-file bench/ds4-2026-10-05/goldens/p600/tokens.i32 \
  -n 128 --slots auto --arena-gib 70 --pcie 0.55 --ppl --dense-requant q6_k \
  --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin > $O/p600-isolated.log 2>&1
echo "$(date '+%F %T') gate: p600 rc=$? $(grep -iE 'ppl|perplex|tok/s' $O/p600-isolated.log | tail -3 | tr '\n' ' ')" | tee -a $O/gate.log
: # calibration stays paused: GPU window for strata-glm (resume with v41calib start)
echo "$(date '+%F %T') gate: done; calibration PAUSED for the strata-glm GPU window" | tee -a $O/gate.log
