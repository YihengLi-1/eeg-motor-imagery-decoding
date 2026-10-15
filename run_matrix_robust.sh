#!/bin/bash
# 杀不死的多 seed 矩阵驱动器
# - caffeinate 防止待机睡眠杀进程
# - 每个配置独立子进程(跑完即释放 MPS 显存,不累积)
# - 结果增量写盘;中途死了重跑本脚本会自动跳过已完成的
# 按"最值钱先跑"排序:fullFT 宽带(78% 那个)最前
cd "$HOME/bci-project" || exit 1
PY="./venv/bin/python"
LOG="$HOME/bci-project/matrix_progress.log"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "===== 驱动器启动 $(date '+%H:%M:%S') =====" | tee -a "$LOG"

# 确保预处理缓存就绪(已存在则秒过)
$PY step3d_matrix.py prep 2>>"$LOG" | tee -a "$LOG"

# (kind band) 配置,按重要性排序
ORDER=(
  "fullFT broad_0.1-75"
  "fullFT narrow_8-30"
  "EEGNet broad_0.1-75"
  "EEGNet narrow_8-30"
  "probe broad_0.1-75"
  "probe narrow_8-30"
)
SEEDS=(0 1 2 3)

# 配置优先:fullFT 宽带的 4 个 seed 先连续跑完,最关键的"78% 稳不稳"最先有答案
for cfg in "${ORDER[@]}"; do
  set -- $cfg
  kind=$1; band=$2
  for seed in "${SEEDS[@]}"; do
    echo "--- [$kind/$band seed$seed] 开始 $(date '+%H:%M:%S') ---" | tee -a "$LOG"
    $PY step3d_matrix.py run "$kind" "$band" "$seed" 2>>"$LOG" | tee -a "$LOG"
    sleep 3   # 让显存彻底释放再开下一个
  done
done

echo "===== 全部配置跑完,汇总 $(date '+%H:%M:%S') =====" | tee -a "$LOG"
$PY step3d_matrix.py summary 2>>"$LOG" | tee -a "$LOG"
echo "===== 驱动器结束 $(date '+%H:%M:%S') ALL_DONE =====" | tee -a "$LOG"
