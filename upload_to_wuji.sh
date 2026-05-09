#!/usr/bin/env bash
# Upload BundleSDF -> root@183.60.152.19:/root/wuji_ws_0/BundleSDF
# 用 rsync 增量同步,断点续传安全,反复跑只传变更。
#
# 用法:
#   ./upload_to_wuji.sh                # 默认: 代码 + .git, 不带 my_data / 样例数据 / checkpoints
#   ./upload_to_wuji.sh --with-data    # 加上 my_data, 2022-*_milk 样例数据, checkpoints
#   ./upload_to_wuji.sh -n             # dry-run, 只看会传什么不真的传
#   ./upload_to_wuji.sh --delete       # 远端删除本地没有的文件 (危险, 先 -n 看一遍)
#   ./upload_to_wuji.sh --with-data -n # 组合用
#
# 首次运行会触发 ssh host key 接受 (StrictHostKeyChecking=accept-new)。
# 不想每次输密码: 装 sshpass (apt/brew install sshpass),
#                或 ssh-copy-id root@183.60.152.19 一次配公钥 (推荐)。

# 服务器启动docker，和本地不一致
# bash /root/wuji_ws_0/BundleSDF/docker/run_container_wuji.sh 

set -euo pipefail

REMOTE_USER=root
REMOTE_HOST=183.60.152.19
REMOTE_DIR=/root/wuji_ws_0/BundleSDF
LOCAL_DIR=/home/l/BundleSDF
PASSWORD=${WUJI_PASSWORD:-}

WITH_DATA=0
DRY_RUN=0
DELETE=0
for arg in "$@"; do
  case "$arg" in
    --with-data) WITH_DATA=1 ;;
    -n|--dry-run) DRY_RUN=1 ;;
    --delete)    DELETE=1 ;;
    -h|--help)   sed -n '1,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

EXCLUDES=(
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ipynb_checkpoints/'
  --exclude='BundleTrack/build/'
  --exclude='build/'
  --exclude='*.egg-info/'
  --exclude='mycuda/*.so'         # 服务器编译产物 cpython-310,别覆盖
  --exclude='mycuda/build/'
  --exclude='docker/dockerfile'   # 服务器版有 7 类网络/编译补丁,别覆盖
  --exclude='build.sh'            # 服务器版有 --no-build-isolation 补丁,别覆盖
  --exclude='wandb/'
  --exclude='.history/'
  --exclude='.vscode/'
  --exclude='.idea/'
  --exclude='*.tar.*'
  --exclude='*.zip'
  --exclude='upload_to_wuji.sh'   # 不把脚本自己传上去,看你需要可去掉
)

if [[ $WITH_DATA -eq 0 ]]; then
  EXCLUDES+=(
    --exclude='my_data/'
    --exclude='2022-*/'
    --exclude='reconstruct_my/checkpoints/'
  )
fi

if [[ $DRY_RUN -eq 1 ]]; then
  # dry-run: 列出每个具体文件 (--itemize-changes), 不要 progress2 (会压总进度)
  RSYNC_FLAGS=(-aHvi -n --human-readable)
else
  # 真传: progress2 看总体进度
  RSYNC_FLAGS=(-aHv --info=progress2 --partial --human-readable)
fi
[[ $DELETE -eq 1 ]] && RSYNC_FLAGS+=(--delete)

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

if [[ -n "$PASSWORD" ]] && command -v sshpass >/dev/null 2>&1; then
  echo "[info] 使用 sshpass 自动输密码 (来自 \$WUJI_PASSWORD)"
  PREFIX=(sshpass -p "$PASSWORD")
else
  if [[ -z "$PASSWORD" ]]; then
    echo "[info] 未设置 \$WUJI_PASSWORD, 走公钥或交互式输密码"
  else
    echo "[info] 已设 \$WUJI_PASSWORD 但未装 sshpass, ssh 仍会交互式提示"
  fi
  echo "[hint] 推荐: 'ssh-copy-id $REMOTE_USER@$REMOTE_HOST' 一次性配公钥(免密最安全)"
  echo "[hint] 或:   'WUJI_PASSWORD=xxx ./upload_to_wuji.sh' + 'sudo apt install sshpass'"
  PREFIX=()
fi

echo "[info] 确保远端目录存在: $REMOTE_DIR"
"${PREFIX[@]}" ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

echo "[info] 开始 rsync   (--with-data=$WITH_DATA  --dry-run=$DRY_RUN  --delete=$DELETE)"
"${PREFIX[@]}" rsync \
  "${RSYNC_FLAGS[@]}" \
  -e "ssh ${SSH_OPTS[*]}" \
  "${EXCLUDES[@]}" \
  "$LOCAL_DIR/" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo "[done] 同步完成 -> $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
