#!/usr/bin/env bash
# 在 Mac 上一次性配置 Cloudflare 固定隧道，并生成 Colab 要读的 secrets.json 和本地的 .env
#
# 用法：
#   ./scripts/setup_tunnel.sh llm.example.com [Drive 里 models 文件夹的路径]
#
# 第二个参数省略时，自动查找 Google Drive 桌面版的 "My Drive/models"（只有一个账号时）。
# 没装 Drive 桌面版：传一个本地目录，之后手动把生成的 secrets.json 上传到 Drive 的 models/ 里。
set -euo pipefail

HOST="${1:?用法: $0 <子域名，如 llm.example.com> [Drive models 目录]}"
TUNNEL_NAME="${TUNNEL_NAME:-colab-llm}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 1. 找到 Drive 里的 models 目录
if [[ $# -ge 2 ]]; then
  DRV="$2"
else
  shopt -s nullglob
  drives=("$HOME"/Library/CloudStorage/GoogleDrive-*)
  if [[ ${#drives[@]} -ne 1 ]]; then
    echo "找到 ${#drives[@]} 个 Google Drive 账号，请用第二个参数指定 models 目录，例如：" >&2
    for d in "${drives[@]}"; do echo "  \"$d/My Drive/models\"" >&2; done
    exit 1
  fi
  DRV="${drives[0]}/My Drive/models"
  echo "使用 Drive 目录: $DRV（必须和 Colab 是同一个 Google 账号）"
fi
mkdir -p "$DRV"

# 2. 安装并登录 cloudflared
command -v cloudflared >/dev/null || brew install cloudflared
if [[ ! -f "$HOME/.cloudflared/cert.pem" ]]; then
  echo "浏览器将打开 Cloudflare 授权页，请选中 ${HOST#*.} 并点 Authorize"
  cloudflared tunnel login
fi

# 3. 创建隧道（已存在则复用）并绑定 DNS
if ! cloudflared tunnel info "$TUNNEL_NAME" >/dev/null 2>&1; then
  cloudflared tunnel create "$TUNNEL_NAME" >/dev/null
  echo "已创建隧道 $TUNNEL_NAME"
fi
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOST" 2>&1 | tail -1 || {
  echo "绑定 DNS 失败：$HOST 可能已有记录。去 Cloudflare 后台确认后再试。" >&2; exit 1; }

# 4. 生成 secrets.json（给 Colab）和 .env（给本地客户端），密钥不打印到屏幕
TOKEN="$(cloudflared tunnel token "$TUNNEL_NAME")"
KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
TOKEN="$TOKEN" KEY="$KEY" HOST="$HOST" python3 - "$DRV/secrets.json" <<'EOF'
import json, os, sys
json.dump({"CF_TUNNEL_TOKEN": os.environ["TOKEN"], "CF_HOSTNAME": os.environ["HOST"],
           "LLM_API_KEY": os.environ["KEY"]}, open(sys.argv[1], "w"), indent=1)
EOF
chmod 600 "$DRV/secrets.json"
printf 'COLAB_LLM_URL=https://%s/v1\nCOLAB_LLM_KEY=%s\nCOLAB_LLM_MODEL=qwen3.8-27b\n' "$HOST" "$KEY" > "$REPO_DIR/.env"
chmod 600 "$REPO_DIR/.env"

echo
echo "完成："
echo "  Colab 读取: $DRV/secrets.json"
echo "  本地客户端: $REPO_DIR/.env（API key 在这里）"
echo "  调用地址:   https://$HOST/v1"
