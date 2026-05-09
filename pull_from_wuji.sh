#!/usr/bin/env bash
# Pull BundleSDF results: root@183.60.152.19:/root/wuji_ws_0/BundleSDF/my_data/<NAME>/results
#                      -> /home/l/BundleSDF/my_data/<NAME>/results
# 用 rsync 增量拉取,断点续传安全,反复跑只传变更。
#
# 用法:
#   ./pull_from_wuji.sh 20260505_220349                       # 拉单个数据的 results
#   ./pull_from_wuji.sh 20260505_220349 20260504_220811       # 拉多个
#   ./pull_from_wuji.sh --all                                 # 拉远端 my_data 下所有 */results
#   ./pull_from_wuji.sh -n 20260505_220349                    # dry-run
#   ./pull_from_wuji.sh --delete 20260505_220349              # 让本地与远端 results 完全一致 (危险, 先 -n 看)
#   ./pull_from_wuji.sh --whole 20260505_220349               # 拉整个数据目录 (不只 results)
#
# 不想每次输密码: 装 sshpass + WUJI_PASSWORD=xxx, 或 ssh-copy-id (推荐)。

set -euo pipefail

REMOTE_USER=root
REMOTE_HOST=183.60.152.19
REMOTE_DIR=/root/wuji_ws_0/BundleSDF
LOCAL_DIR=/home/l/BundleSDF
PASSWORD=${WUJI_PASSWORD:-}

ALL=0
DRY_RUN=0
DELETE=0
WHOLE=0
NAMES=()
for arg in "$@"; do
  case "$arg" in
    --all)        ALL=1 ;;
    -n|--dry-run) DRY_RUN=1 ;;
    --delete)     DELETE=1 ;;
    --whole)      WHOLE=1 ;;
    -h|--help)    sed -n '1,18p' "$0"; exit 0 ;;
    -*)           echo "unknown flag: $arg"; exit 1 ;;
    *)            NAMES+=("$arg") ;;
  esac
done

if [[ $ALL -eq 0 && ${#NAMES[@]} -eq 0 ]]; then
  echo "用法: $0 <data_name> [<data_name> ...]   或   $0 --all"
  echo "例:   $0 20260505_220349"
  exit 1
fi

EXCLUDES=(
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.ipynb_checkpoints/'
  --exclude='wandb/'
)

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_FLAGS=(-aHvi -n --human-readable)
else
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
  PREFIX=()
fi

# --all: 远端列出 my_data/* 目录名
if [[ $ALL -eq 1 ]]; then
  echo "[info] 远端枚举 $REMOTE_DIR/my_data/*"
  REMOTE_LIST=$("${PREFIX[@]}" ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "ls -1 $REMOTE_DIR/my_data 2>/dev/null | grep -v '^$'")
  if [[ -z "$REMOTE_LIST" ]]; then
    echo "[warn] 远端 my_data 为空"
    exit 0
  fi
  while IFS= read -r line; do
    NAMES+=("$line")
  done <<< "$REMOTE_LIST"
fi

# 带重试: 区分 ssh 自身错(255, 网络/握手问题, 重试) 和 远端命令真失败(其它码, 不重试)
# 用法: ssh_retry "<remote shell command>"  -> 透传远端命令退出码; ssh 自身错重试到上限后退出
SSH_MAX_RETRY=5
SSH_RETRY_SLEEP=3
ssh_retry() {
  local cmd="$1"
  local i rc
  for ((i=1; i<=SSH_MAX_RETRY; i++)); do
    set +e
    "${PREFIX[@]}" ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" "$cmd"
    rc=$?
    set -e
    if [[ $rc -ne 255 ]]; then
      return $rc
    fi
    echo "[warn] ssh 连接失败(255), 第 $i/$SSH_MAX_RETRY 次, ${SSH_RETRY_SLEEP}s 后重试" >&2
    sleep $SSH_RETRY_SLEEP
  done
  echo "[error] ssh 连接连续失败 $SSH_MAX_RETRY 次, 放弃" >&2
  return 255
}

# rsync 也可能因 ssh 闪断挂掉, 同样重试
RSYNC_MAX_RETRY=5
rsync_retry() {
  local src="$1" dst="$2"
  local i rc
  for ((i=1; i<=RSYNC_MAX_RETRY; i++)); do
    set +e
    "${PREFIX[@]}" rsync \
      "${RSYNC_FLAGS[@]}" \
      -e "ssh ${SSH_OPTS[*]}" \
      "${EXCLUDES[@]}" \
      "$src" "$dst"
    rc=$?
    set -e
    # 0 ok; 24 vanished files (无害); 其他都重试
    if [[ $rc -eq 0 || $rc -eq 24 ]]; then
      return 0
    fi
    echo "[warn] rsync 失败(rc=$rc), 第 $i/$RSYNC_MAX_RETRY 次, ${SSH_RETRY_SLEEP}s 后重试" >&2
    sleep $SSH_RETRY_SLEEP
  done
  echo "[error] rsync 连续失败 $RSYNC_MAX_RETRY 次, 放弃: $src -> $dst" >&2
  return 1
}

echo "[info] 拉取条目: ${NAMES[*]}"
echo "[info] 模式: WHOLE=$WHOLE  DRY_RUN=$DRY_RUN  DELETE=$DELETE"

FAILED=()
for name in "${NAMES[@]}"; do
  if [[ $WHOLE -eq 1 ]]; then
    SUBPATH="my_data/$name/"
  else
    SUBPATH="my_data/$name/results/"
  fi

  REMOTE_PATH="$REMOTE_DIR/$SUBPATH"
  LOCAL_PATH="$LOCAL_DIR/$SUBPATH"

  # 远端存在性检查 (区分 ssh 错和真不存在)
  set +e
  ssh_retry "test -d $REMOTE_PATH"
  rc=$?
  set -e
  case $rc in
    0)   ;;  # 存在
    1)   echo "[skip] 远端不存在: $REMOTE_PATH"; continue ;;
    255) echo "[fail] ssh 拒连, 跳过: $name"; FAILED+=("$name"); continue ;;
    *)   echo "[fail] 远端检查异常 rc=$rc: $name"; FAILED+=("$name"); continue ;;
  esac

  mkdir -p "$LOCAL_PATH"
  echo "[sync] $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH  ->  $LOCAL_PATH"
  if ! rsync_retry "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"; then
    FAILED+=("$name")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[done] 完成,但有失败项: ${FAILED[*]}"
  exit 1
fi
echo "[done] 拉取完成"
