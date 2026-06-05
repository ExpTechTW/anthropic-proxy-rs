#!/bin/bash
input=$(cat)

# ── 標籤（純文字，任何字型都能正常顯示）──
# 若日後想改用 Nerd Fonts 圖示，請用 UTF-8 八進位位元組（printf 的 \u 跳脫多數不支援）：
#   ICON_CTX=$(printf '\357\213\233')   # U+F2DB 晶片
LABEL_CTX="ctx"

# ── 進度條寬度（格數）──
# Claude Code 傳給狀態列的 JSON「沒有」終端寬度欄位,所以無法自動撐滿整行;
# 這裡用固定較寬的格數把空間填得更滿。想更寬/更窄改這個數字即可。
BAR_WIDTH=20

# ── 費率（NTD / 百萬 token）──
RATE_IN=90      # 輸入(含寫入快取的 token)
RATE_OUT=470    # 輸出
RATE_CACHE=7    # 讀取快取(折扣價)

# ── 讀取資料 ──
model=$(echo "$input" | jq -r '.model.display_name // ""')
ctx_used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
ctx_in=$(echo "$input" | jq -r '.context_window.total_input_tokens // empty')
ctx_size=$(echo "$input" | jq -r '.context_window.context_window_size // empty')

# 分母改用「自動壓縮視窗」，對齊 /context。Claude Code 的 used_percentage /
# context_window_size 用的是模型「完整」視窗(200K/1M),跟壓縮門檻是解耦的,
# 所以後端視窗較小時,狀態列比例會比 /context 小很多。優先吃 CLAUDE_CODE_AUTO_COMPACT_WINDOW
# 環境變數;沒有就退回回報的 context_window_size。
ctx_window="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-$ctx_size}"

# ── 顏色函式：依剩餘百分比決定顏色 ──
# 綠色 ≥50% │ 黃色 21-49% │ 紅色 ≤20%
color_by_remain() {
  local val=$1
  if [ "$val" -le 20 ]; then
    printf '\033[31m'   # 紅色
  elif [ "$val" -le 49 ]; then
    printf '\033[33m'   # 黃色
  else
    printf '\033[32m'   # 綠色
  fi
}

# ── 進度條：BAR_WIDTH 格寬度 ──
mini_bar() {
  local percent=$1
  local width=$BAR_WIDTH
  local filled=$((percent * width / 100))
  local empty=$((width - filled))
  local i=0
  while [ $i -lt $filled ]; do printf '━'; i=$((i + 1)); done
  while [ $i -lt $width ]; do printf '┄'; i=$((i + 1)); done
}

# ── 分隔符號 ──
SEP=$(printf '\033[2m │ \033[0m')

# ── 組合輸出 ──
parts=""

# Model（紫色粗體，讓它跟其他資訊有區隔）
if [ -n "$model" ]; then
  parts=$(printf '\033[1;35m%s\033[0m' "$model")
fi

# Context 已用（進度條依「已用」填滿；顏色依「剩餘」決定）
# 優先用「token 數 ÷ 自動壓縮視窗」自行計算(對齊 /context);
# 缺 token 數時才退回 Claude Code 的 used_percentage(完整視窗基準)。
used=""
if [ -n "$ctx_in" ] && [ -n "$ctx_window" ] && [ "$ctx_window" -gt 0 ] 2>/dev/null; then
  used=$(( ctx_in * 100 / ctx_window ))
elif [ -n "$ctx_used" ]; then
  used=$(printf '%.0f' "$ctx_used")
fi
# 夾在 0–100,避免估算誤差讓進度條溢出
if [ -n "$used" ]; then
  [ "$used" -lt 0 ] 2>/dev/null && used=0
  [ "$used" -gt 100 ] 2>/dev/null && used=100
fi
# 把實際使用的視窗格式化成 k（例如 105120 → 105k），方便確認讀到的是哪個值：
# 105k = 有吃到 CLAUDE_CODE_AUTO_COMPACT_WINDOW；1000k/200k = 退回模型完整視窗
win_label=""
if [ -n "$ctx_window" ] && [ "$ctx_window" -gt 0 ] 2>/dev/null; then
  win_label="$(( ctx_window / 1000 ))k"
fi
if [ -n "$used" ]; then
  remain=$((100 - used))
  color=$(color_by_remain "$remain")
  bar=$(mini_bar "$used")
  ctx_str=$(printf '%s%s %s %s%%\033[0m' "$color" "$LABEL_CTX" "$bar" "$used")
  if [ -n "$win_label" ]; then
    ctx_str="$ctx_str$(printf ' \033[2m%s\033[0m' "$win_label")"
  fi
  parts="$parts$SEP$ctx_str"
fi

# ── 累計費用：直接從本 session 的 transcript 算 ──
# transcript_path 是這個 session 的對話紀錄(~/.claude*/projects/.../<session>.jsonl),
# 每則 assistant 訊息的 message.usage 都有 token 數。直接加總 → 本來就「跟著 session」:
# session 被刪(transcript 檔移除)時費用自然歸零,不會留下孤兒狀態檔。
# transcript 是 append-only,所以用「檔案大小+mtime」當鍵做快取,沒新增就不重算整個檔
# (快取放 /tmp,只是加速;就算殘留也無害、可隨時重算)。
transcript=$(echo "$input" | jq -r '.transcript_path // empty')
session=$(echo "$input" | jq -r '.session_id // "default"')
cost_ntd=""
if [ -n "$transcript" ] && [ -f "$transcript" ]; then
  fp=$(stat -f '%z-%m' "$transcript" 2>/dev/null || stat -c '%s-%Y' "$transcript" 2>/dev/null)
  cache="/tmp/cc-cost-${session//[^A-Za-z0-9_-]/_}.cache"
  c_fp=""; c_val=""
  [ -f "$cache" ] && read -r c_fp c_val < "$cache"
  if [ -n "$fp" ] && [ "$fp" = "$c_fp" ]; then
    cost_ntd="$c_val"
  else
    cost_ntd=$(jq -r --argjson ri "$RATE_IN" --argjson ro "$RATE_OUT" --argjson rc "$RATE_CACHE" '
        select(.type=="assistant") | .message.usage // empty
        | ((.input_tokens // 0) + (.cache_creation_input_tokens // 0)) * $ri
          + (.output_tokens // 0) * $ro
          + (.cache_read_input_tokens // 0) * $rc
      ' "$transcript" 2>/dev/null | awk '{ s += $1 } END { printf "%.2f", s / 1000000 }')
    printf '%s %s\n' "$fp" "$cost_ntd" > "$cache.tmp" 2>/dev/null && mv "$cache.tmp" "$cache" 2>/dev/null
  fi
fi
if [ -n "$cost_ntd" ] && [ "$cost_ntd" != "0.00" ]; then
  parts="$parts$SEP$(printf '\033[36mNT$%s\033[0m' "$cost_ntd")"
fi

printf '%s' "$parts"
