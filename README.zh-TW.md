# anthropic-proxy-rs

[English](README.md) · **繁體中文**

[![CI & Release](https://github.com/ExpTechTW/anthropic-proxy-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ExpTechTW/anthropic-proxy-rs/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/ExpTechTW/anthropic-proxy-rs?sort=date)](https://github.com/ExpTechTW/anthropic-proxy-rs/releases)

高效能 Rust 代理,把 **Anthropic Messages API** 即時翻譯成 **OpenAI Chat Completions** 格式。讓 Claude Code、Claude Desktop 或任何 Anthropic API 客戶端,直接連到 OpenRouter、原生 OpenAI、Azure OpenAI,或任何 OpenAI 相容端點(vLLM、Ollama、LM Studio、自架 gateway……)。

> 本專案 fork 自 [m0n0x41d/anthropic-proxy-rs](https://github.com/m0n0x41d/anthropic-proxy-rs),由 [ExpTech](https://github.com/ExpTechTW) 維護。新增 `tool_choice` / `metadata` / `refusal` 支援、`count_tokens` 端點、保留上游狀態碼,以及連線穩定性強化。

## 特色

- **快速輕量** — Rust 非同步 I/O(約 3 MB 執行檔)
- **完整串流** — Server-Sent Events(SSE)即時回應
- **工具呼叫** — function/tool calling,含 `tool_choice`
- **通用** — 任何 OpenAI 相容 API(OpenRouter、OpenAI、Azure、本機 LLM)
- **延伸思考** — 偵測 Claude 的 reasoning 模式並據此切換模型
- **自我學習技能**(選用,預設關閉)— 從服務過的對話學到可重用的教訓,經 web 查證與信任分層,把相關的注入後續請求(Qdrant 儲存、不需 fine-tune)。見[自我學習技能](#自我學習技能)
- **穩定可靠** — 暫時性錯誤自動重試、保留上游狀態碼、600 秒請求逾時
- **即插即用** — 相容官方 Anthropic SDK 與 Claude Code

## 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/v1/messages` | 主要對話端點(串流 + 非串流);也接受結尾斜線 |
| `POST` | `/v1/messages/count_tokens` | Token 計數 —— 啟用時透過上游 `/tokenize` 精確計算,否則用本地 BPE 估算 |
| `GET`  | `/v1/models` | 列出上游回報的模型,翻譯成 Anthropic 格式 |
| `GET`  | `/health` | 存活檢查(回 `OK`) |
| `GET`  | `/metrics` | Prometheus 指標 |

## 下載

預編譯二進制發佈於 [GitHub Releases](https://github.com/ExpTechTW/anthropic-proxy-rs/releases),提供:

| 平台 | 檔案 |
|------|------|
| Linux x86_64(靜態 musl) | `anthropic-proxy-x86_64-unknown-linux-musl.tar.gz` |
| Linux arm64(靜態 musl) | `anthropic-proxy-aarch64-unknown-linux-musl.tar.gz` |
| macOS arm64(Apple Silicon) | `anthropic-proxy-aarch64-apple-darwin.tar.gz` |

```bash
# Linux x86_64 —— 最新正式版
curl -fsSL https://github.com/ExpTechTW/anthropic-proxy-rs/releases/latest/download/anthropic-proxy-x86_64-unknown-linux-musl.tar.gz \
  | tar -xz && ./anthropic-proxy --version
```

每個檔案都附對應的 `.sha256` 供驗證。Linux 二進制為靜態連結(musl),可在任何發行版執行。

**發佈管道與版本號** —— 版本採日期格式:`vYYYY.MM.DD+build.N`,其中 `N` 每個 UTC 日歸零為 `1`。

- 推送到 **`main`** 會發佈 **預發佈版**(`… Pre-release (auto)`)。
- 推送到 **`release`** 會發佈 **正式版**(`… Release (auto)`)—— `releases/latest` 指向的就是這個。

## 快速開始

```bash
# 安裝 Rust(若尚未安裝)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 編譯並執行

```bash
cargo build --release
UPSTREAM_BASE_URL=https://api.openai.com \
UPSTREAM_API_KEY=sk-... \
./target/release/anthropic-proxy
```

### 以 Cargo 安裝

```bash
cargo install --git https://github.com/ExpTechTW/anthropic-proxy-rs --locked
```

### 安裝後隨處執行

```bash
UPSTREAM_BASE_URL=https://openrouter.ai/api \
UPSTREAM_API_KEY=sk-or-... \
anthropic-proxy
```

### Docker

repo 的 [`Dockerfile`](Dockerfile) 會**下載預編譯 release 二進制** —— 不需 Rust 工具鏈,建置很快。用 build args 選擇管道:

```bash
# 最新正式版(預設)
docker build -t anthropic-proxy .

# 最新預發佈版(main 分支建置)
docker build -t anthropic-proxy --build-arg CHANNEL=prerelease .

# 釘特定 tag(覆寫 CHANNEL;預發佈版也適用)
docker build -t anthropic-proxy --build-arg VERSION=v2026.06.05+build.3 .

# 多架構
docker buildx build --platform linux/amd64,linux/arm64 -t anthropic-proxy .

docker run -p 3000:3000 \
  -e UPSTREAM_BASE_URL=https://openrouter.ai/api \
  -e UPSTREAM_API_KEY=sk-or-... \
  anthropic-proxy
```

| Build arg | 預設 | 意義 |
|-----------|------|------|
| `CHANNEL` | `release` | `release` = 最新正式版 · `prerelease` = 最新預發佈版 |
| `VERSION` | (空) | 釘特定 tag,例如 `v2026.06.05+build.3`(覆寫 `CHANNEL`) |

若要改成從原始碼編譯,使用 [`Dockerfile.source`](Dockerfile.source):

```bash
docker build -f Dockerfile.source -t anthropic-proxy .
```

## 搭配 Claude Code

```bash
anthropic-proxy --daemon && ANTHROPIC_BASE_URL=http://localhost:3000 claude
```

### 建議設定 — [`examples/claude-code-settings.json`](examples/claude-code-settings.json)

複製到 `~/.claude/settings.json`(或專案層級的 `.claude/settings.json`),填入你的代理 URL 與金鑰。真正關鍵的只有一個:

**`CLAUDE_CODE_AUTO_COMPACT_WINDOW`** —— 設成上游模型的**真實** context 視窗(token 數,例如 `105120`)。當較小的後端被對映到 `claude-sonnet-4-5` 這種名稱時,Claude Code 會以為它是模型完整的 200K/1M 視窗,於是**遠遠超過**真實上限才自動壓縮 —— 最後每個請求都 `context length exceeded` 失敗。

> ⚠️ **別用 `CLAUDE_CODE_MAX_CONTEXT_TOKENS`** —— 依[官方文件](https://code.claude.com/docs/en/env-vars)它只有搭配 `DISABLE_COMPACT` 才生效。驅動自動壓縮的是 `CLAUDE_CODE_AUTO_COMPACT_WINDOW`。
>
> **不需要 `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`** —— Claude Code 保留固定 ~33K token 緩衝,所以較小的視窗本來就會留餘裕就壓縮(105K 約在 ~68%)。那個 override 只能讓壓縮**更早**(設高於預設會被忽略),無法延後。

用 `/context` 驗證:應顯示真實視窗(例如 `30.6k / 105.1k`)與一行 `Auto-compact window`。

### 狀態列 — [`examples/statusline.sh`](examples/statusline.sh)

搭配用的狀態列,顯示**真實**的 context 比例(對齊 `/context`,而非 Claude Code 除以完整視窗的 `used_percentage`)與**累計花費**:

```text
Sonnet 4.5 │ ctx ━━━━━┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ 29% 105k │ NT$6.38
```

費用直接從 session 的 transcript 加總,所以**跟著 session**:session 一刪就消失,不留孤兒狀態檔。安裝(需要 `jq`),再到腳本頂端設定你 gateway 的費率:

```bash
cp examples/statusline.sh ~/.claude/statusline.sh && chmod +x ~/.claude/statusline.sh
```

## 設定

### 命令列選項

```bash
anthropic-proxy --help
```

**指令**

| 指令 | 說明 |
|------|------|
| `stop` | 停止執行中的 daemon |
| `status` | 檢查 daemon 狀態 |

**選項**

| 選項 | 縮寫 | 說明 |
|------|------|------|
| `--config <FILE>` | `-c` | 自訂 `.env` 檔路徑 |
| `--debug` | `-d` | 開啟除錯日誌 |
| `--verbose` | `-v` | 詳細日誌(會記錄完整請求/回應內容) |
| `--port <PORT>` | `-p` | 監聽埠(覆寫 `PORT`) |
| `--bind <ADDR>` | | 監聽位址(覆寫 `ANTHROPIC_PROXY_BIND`,預設 `0.0.0.0`) |
| `--system-prompt-ignore <TEXT>` | | 轉發前移除 system prompt 詞彙(可重複,或用 `;` 分隔) |
| `--daemon` | | 以背景 daemon 執行 |
| `--pid-file <FILE>` | | PID 檔路徑(預設 `/tmp/anthropic-proxy.pid`) |
| `--help` | `-h` | 顯示說明 |
| `--version` | `-V` | 顯示版本 |

### 環境變數

可透過環境變數或 `.env` 檔設定:

| 變數 | 必填 | 預設 | 說明 |
|------|------|------|------|
| `UPSTREAM_BASE_URL` | **是** | – | OpenAI 相容端點 URL |
| `UPSTREAM_API_KEY` | 否\* | – | 上游服務的 API key |
| `UPSTREAM_API_KEY_PASSTHROUGH` | 否 | `false` | 從每個請求的 `x-api-key` 標頭取得 key(`true`/`false`) |
| `PORT` | 否 | `3000` | 伺服器埠 |
| `ANTHROPIC_PROXY_BIND` | 否 | `0.0.0.0` | 綁定位址。設 `127.0.0.1` 可限制只在本機。綁 `0.0.0.0` 會記錄警告。 |
| `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS` | 否 | – | 轉發前要移除的 system prompt 詞彙(`;` 或換行分隔) |
| `ANTHROPIC_PROXY_MODEL_MAP` | 否 | – | 上游呼叫前的精確模型對映(`source=target;other=target`) |
| `ANTHROPIC_PROXY_UPSTREAM_TOKENIZE` | 否 | `false` | 用上游 vLLM 風格的 `/tokenize` 取得**精確**的 `count_tokens` 與準確的溢出夾值,而非本地估算 |
| `ANTHROPIC_PROXY_EFFORT_MAP` | 否 | – | 把 Anthropic thinking 對映成 OpenAI 的 `reasoning_effort` 欄位(見下方) |
| `REASONING_MODEL` | 否 | (請求模型) | 啟用延伸思考時使用的模型\*\* |
| `COMPLETION_MODEL` | 否 | (請求模型) | 一般請求(無思考)使用的模型\*\* |
| `DEBUG` | 否 | `false` | 除錯日誌(`1` 或 `true`) |
| `VERBOSE` | 否 | `false` | 詳細日誌(`1` 或 `true`) |

\* 若上游端點需要驗證則為必填。
\*\* 代理會偵測請求是否啟用延伸思考(`thinking` 參數),若是則路由到 `REASONING_MODEL`;沒有思考的請求使用 `COMPLETION_MODEL`。兩者皆未設定時,使用客戶端請求中的模型。`ANTHROPIC_PROXY_MODEL_MAP` 會在這個選擇**之後**才套用。

`UPSTREAM_BASE_URL` 接受以下任一形式:
- 服務基底 URL:`https://api.openai.com` → `/v1/chat/completions`
- 帶版本的基底 URL:`https://gateway.company.internal/v2` → `/v2/chat/completions`
- 完整端點:`https://gateway.company.internal/v2/chat/completions` → 原樣使用

### Reasoning effort(思考強度)

`ANTHROPIC_PROXY_EFFORT_MAP` 會依客戶端的 thinking 請求,對上游送出 OpenAI 的 `reasoning_effort` 欄位(相容的 gateway 會把等級解析成每模型的 thinking budget) —— 一組全域門檻,加上可選的 per-(client-)model 覆寫,每個 tier 是 `effort:maxBudget`:

```bash
ANTHROPIC_PROXY_EFFORT_MAP="low:2048,medium:8192,high:16384;claude-haiku-3-5=low:512,high:16384"
```

- 客戶端的 `thinking.budget_tokens`(夾到 **16384**)會選「第一個 `maxBudget ≥ budget` 的 tier」;全部超過 → 取最高 tier。
- `;` 分段中**沒有** `=` 的是全域預設;`model=tiers` 覆寫單一 client 模型。
- 客戶端若直接帶 Anthropic `effort` 欄位則原樣轉送。等級一律原樣傳出 —— 驗證與「未知/空值的預設」交給上游處理(相容的 gateway 會預設成 `medium`)。
- 只列上游接受的 tier(例如 qwen 吃 `low`/`medium`/`high`;若不支援 `xhigh`/`max` 會 `400`)。未設定時不送 `effort`。

### 設定檔搜尋順序

代理會依序尋找 `.env` 檔:

1. `--config` 指定的自訂路徑
2. 目前工作目錄(`./.env`)
3. 使用者家目錄(`~/.anthropic-proxy.env`)
4. 系統層級(`/etc/anthropic-proxy/.env`)

都找不到時,使用 shell 的環境變數。

### API key 透傳(passthrough)

設定 `UPSTREAM_API_KEY_PASSTHROUGH=true` 後,代理會讀取每個請求的 `x-api-key` 標頭(Anthropic 客戶端的標準標頭),並以 `Authorization: Bearer {key}` 轉發給上游 —— 讓每個客戶端用自己的 key 驗證,而非共用單一的 `UPSTREAM_API_KEY`。

```bash
# 透傳模式(不可設定 UPSTREAM_API_KEY)
UPSTREAM_API_KEY_PASSTHROUGH=true \
UPSTREAM_BASE_URL=https://openrouter.ai/api \
anthropic-proxy
```

限制:
- 不可與 `UPSTREAM_API_KEY` 並用 —— 兩者皆設定時代理會拒絕啟動。
- 若開啟透傳但請求沒有(或為空的)`x-api-key`,則不會送出 `Authorization` 標頭,由上游決定是否接受。

### 模型對映

```bash
ANTHROPIC_PROXY_MODEL_MAP='claude-opus-4-7=openai/gpt-4.1;claude-haiku-3-5=openai/gpt-4.1-mini'
```

### System prompt 清理

轉發前移除指定詞彙(當 gateway/WAF 會擋特定字串時很有用):

```bash
ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS='rm -rf;git reset --hard' anthropic-proxy
# 或
anthropic-proxy --system-prompt-ignore 'rm -rf' --system-prompt-ignore 'git reset --hard'
```

### 以 daemon 執行

```bash
anthropic-proxy --daemon            # 啟動
anthropic-proxy status              # 檢查
anthropic-proxy stop                # 停止
tail -f /tmp/anthropic-proxy.log    # 日誌
```

## 自我學習技能

選用、預設關閉的一層,讓 proxy **從服務過的對話學到可重用的教訓,並把相關的學習知識注入後續請求**——對 client 透明,且全程 best-effort,任何失敗都不會中斷被代理的請求。靈感來自技能庫／經驗學習型 agent(Voyager、ReasoningBank、ExpeL)。

知識存在外部 **[Qdrant](https://qdrant.tech)** 向量庫(不需 fine-tune),每筆都帶**信任分層**——只有 `verified`/`trusted` 會被注入。流程:

1. **注入**(請求路徑上)— 把使用者最後一則訊息做 embedding,取出最相關的前 k 筆 `verified`/`trusted` 技能,以 system block 附加;注入的技能 id 會回在 `x-injected-skills` 標頭。功能開啟時**預設注入**;client 可用請求標頭 **`x-skills-inject: off`**(或 `false`/`0`/`no`)對單一請求關閉。
2. **蒸餾**(背景)— 對話後由 LLM 評判結果(成功/失敗,從嚴:沒有抱怨**不等於**成功),萃取 ≤3 條通用、可重用的教訓(成功與失敗都學),寫成 `candidate`(不可注入)。
3. **驗證**(背景迴圈)— 每個 candidate 由**隔離的**讀取器(把搜尋結果當不可信資料、無工具)對開放網路查證;升級為 `verified` 需要正面判定**且**多個獨立來源佐證(而非模型自信),再經 soak 期升 `trusted`。
4. **策展**(背景迴圈)— 淘汰過保留期的未驗證 candidate、合併近似重複,使知識庫精簡高訊號。
5. **主動學習**(背景迴圈,選用)— 記錄使用者實際問的問題,定期上網研究並蒸餾成 candidate,讓知識庫往真正需要的方向長。

整套是非參數化的——「學習」就是寫 row 進 Qdrant——而且所有背景工作都在請求路徑外,正常請求的延遲與穩定性完全不受影響。

**前置需求:** 可連線的 Qdrant、一個 OpenAI 相容的 **embeddings** 端點(例如以 [llama.cpp](https://github.com/ggml-org/llama.cpp) 或 Ollama 服務的小型多語模型),以及背景學習 LLM 的 chat 端點(預設用上游;把 `ANTHROPIC_PROXY_SKILLS_LLM_URL` 指向免認證的內部 backend,可避免耗用 client key／用戶額度)。驗證沿用內建的 open-websearch。

| 變數 | 預設 | 說明 |
|------|------|------|
| `ANTHROPIC_PROXY_SKILLS_ENABLED` | `false` | **注入**(讀取路徑)總開關 |
| `ANTHROPIC_PROXY_SKILLS_LEARN` | `false` | 啟用**學習**背景迴圈(蒸餾＋驗證＋策展) |
| `ANTHROPIC_PROXY_SKILLS_PROACTIVE` | `false` | 啟用**主動**研究被問的問題(需 `LEARN`) |
| `ANTHROPIC_PROXY_SKILLS_QDRANT_URL` | `http://qdrant:6333` | Qdrant base URL |
| `ANTHROPIC_PROXY_SKILLS_COLLECTION` | `skills` | Qdrant collection 名稱 |
| `ANTHROPIC_PROXY_SKILLS_EMBED_URL` | 上游 `/embeddings` | OpenAI 相容 embeddings 端點 |
| `ANTHROPIC_PROXY_SKILLS_EMBED_MODEL` | – | embedding 模型名(空字串停用檢索) |
| `ANTHROPIC_PROXY_SKILLS_LLM_URL` | 上游 chat URL | 背景學習呼叫的 chat 端點;設定後以**免認證**呼叫 |
| `ANTHROPIC_PROXY_SKILLS_LLM_MODEL` | `auto` | 背景學習／評判用模型 |
| `ANTHROPIC_PROXY_SKILLS_API_KEY` | – | 非請求觸發的背景呼叫用 key(否則回退到最近一次 client key) |
| `ANTHROPIC_PROXY_SKILLS_TOP_K` | `3` | 每次請求最多注入幾筆 |
| `ANTHROPIC_PROXY_SKILLS_MIN_SCORE` | `0.5` | 注入的最低 cosine 分數(過濾弱匹配;bge-m3 建議約 `0.45`) |
| `ANTHROPIC_PROXY_SKILLS_INJECT_TIERS` | `verified,trusted` | 可被注入的層級(排除 candidate) |
| `ANTHROPIC_PROXY_SKILLS_VERIFY_INTERVAL_SECS` | `300` | 驗證迴圈間隔 |
| `ANTHROPIC_PROXY_SKILLS_SOAK_SECS` | `1209600` | `verified` → `trusted` 的 soak 期(預設 14 天) |
| `ANTHROPIC_PROXY_SKILLS_CURATE_INTERVAL_SECS` | `600` | 策展迴圈間隔 |
| `ANTHROPIC_PROXY_SKILLS_PROACTIVE_INTERVAL_SECS` | `600` | 主動學習迴圈間隔 |
| `ANTHROPIC_PROXY_SKILLS_RETENTION_DAYS` | `30` | 淘汰超過此天數的未驗證 candidate |

> **安全。** 從開放網路學習是已知的中毒風險,所以信任分層是承重控制:未驗證知識絕不注入、升級需多個獨立來源佐證(而非模型自信)、讀網頁的 LLM 被隔離(無工具／無寫入)。請把你的 embeddings／LLM／Qdrant 端點視為可信基礎設施。

範例(Docker Compose)— 同址 Qdrant + llama.cpp embedding server,學習導向免認證的內部 backend:

```yaml
  qdrant:
    image: qdrant/qdrant
  embeddings:                       # OpenAI 相容 /v1/embeddings(多語 bge-m3)
    image: ghcr.io/ggml-org/llama.cpp:server
    command: ["-hf","gpustack/bge-m3-GGUF:Q4_K_M","--embeddings","--pooling","cls","-c","8192","-ub","8192","--host","0.0.0.0","--port","8080"]
    # -ub 8192 讓它能把整段輸入塞進一個 physical batch 做 embedding;沒設的話超過預設 512 token 會回 500。
  anthropic-proxy:
    environment:
      ANTHROPIC_PROXY_SKILLS_ENABLED: "true"
      ANTHROPIC_PROXY_SKILLS_LEARN: "true"
      ANTHROPIC_PROXY_SKILLS_PROACTIVE: "true"
      ANTHROPIC_PROXY_SKILLS_QDRANT_URL: "http://qdrant:6333"
      ANTHROPIC_PROXY_SKILLS_EMBED_URL: "http://embeddings:8080/v1/embeddings"
      ANTHROPIC_PROXY_SKILLS_EMBED_MODEL: "bge-m3"
      ANTHROPIC_PROXY_SKILLS_LLM_URL: "http://your-backend:8000/v1/chat/completions"
      ANTHROPIC_PROXY_SKILLS_LLM_MODEL: "your-model"
      ANTHROPIC_PROXY_SKILLS_MIN_SCORE: "0.45"
```

## 支援功能

✅ 文字訊息
✅ System prompt(單一與多個)
✅ 圖片內容(base64)
✅ 工具/函式呼叫 + 工具結果(多輪 `tool_use` ↔ `tool_result` 往返)
✅ `tool_choice` —— `auto` / `any` / `tool` / `none`,含 `disable_parallel_tool_use`
✅ 串流回應(SSE;支援 `\n\n` 與 `\r\n\r\n` 切幀)
✅ 延伸思考(自動模型路由;串流與非串流都保留 `reasoning_content`)
✅ `metadata.user_id`(轉成 OpenAI 的 `user`)
✅ `refusal` 停止原因(由上游 `content_filter` 對映)
✅ 停止序列、`max_tokens`、`temperature`、`top_p`
✅ `POST /v1/messages/count_tokens` —— 啟用 `ANTHROPIC_PROXY_UPSTREAM_TOKENIZE=true` 時透過上游 `/tokenize` 精確計算,否則用本地 BPE 估算;也接受 `?beta=true`
✅ 串流 usage —— 真實的 `input_tokens` / `output_tokens` / `cache_read_input_tokens` 會在 `message_delta` 回報(從上游最後的 usage chunk 擷取),並在 `message_start` 先給一個估算值
✅ Prompt 快取 token 計量 —— 上游的 `prompt_tokens_details.cached_tokens` 會回報為 Anthropic 的 `cache_read_input_tokens`(並從 `input_tokens` 扣除),讓 Claude Code 的快取/成本統計正確

> `top_k` 會被接受但不會轉發 —— Chat Completions 沒有對應參數。
>
> **注意:** 請確認上游模型支援工具使用,尤其是用於 Claude Code 這類程式碼代理時。

## 穩定性

- **保留上游狀態碼。** 客戶端錯誤(`400` 請求錯誤、`401`/`403`/`404`、`413`、`429` 限流)會以原始狀態碼與 Anthropic 格式錯誤內容回傳 —— `{"type":"error","error":{"type":...,"message":...}}` —— 而不是一律被遮成 `502`。只有真正的傳輸失敗(沒有 HTTP 回應)才會對映成 `502`。
- **暫時性失敗自動重試。** 連線/逾時/讀取 body 錯誤,以及可重試狀態(`429`、`5xx`),每個上游 URL 最多以全新連線重試 3 次。
- **Context 溢出自動回復。** 若上游因 `input + max_tokens` 超過模型 context 視窗而拒絕,代理會重新 tokenize 實際請求以取得真實輸入大小(並取「它」與「錯誤訊息自身的下界」之中較大者,確保一定收斂),把 `max_tokens` 夾到剛好放得下並重試 —— 讓「剛好超過上限」的對話(以及 Claude Code 的 `/compact`,它本身也要求輸出而會卡死)還是能完成,而非直接 `400` 失敗。
- **過期連線強化。** 連線池的閒置連線維持短壽命並開啟 TCP keep-alive,消除「重用被上游靜默關閉的 socket」造成的間歇性 `502`。
- **600 秒請求逾時**,與前方常見的 nginx `proxy_read_timeout` 對齊,長生成與串流不會被提前截斷。

## 置於反向代理(nginx)之後

當本代理跑在 nginx/OpenResty 之後,對應的 location 建議如下:

```nginx
location / {
    proxy_pass http://anthropic-proxy:3000;

    # SSE 串流
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_request_buffering off;

    # 讓代理的 JSON 錯誤內容原樣回傳
    proxy_intercept_errors off;

    # 與代理的 600 秒請求逾時對齊
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    send_timeout 600s;
}
```

- 保持 **`proxy_intercept_errors off`**,讓代理的 Anthropic 格式 JSON 錯誤能回到客戶端(否則 `error_page` 會把它換成純文字)。
- 串流務必 **關閉緩衝**。
- 逾時對齊 **600 秒**(或同步調整代理),避免任一端截斷長回應。

## 已知限制

以下 Anthropic API 功能**目前不支援**(Claude Code 等工具沒有這些參數也能正常運作):

- `service_tier` 參數(沒有可移植的 OpenAI 相容對應)
- `context_management` 參數(Anthropic 伺服器端功能;上游無對應)
- `container` 參數(Anthropic 程式碼執行沙箱;上游無對應)
- 回應中的 Citations
- `pause_turn` 停止原因(僅由 Anthropic 伺服器端工具循環產生,本代理不轉發)
- Message Batches API
- Files API
- Admin API

未啟用 `ANTHROPIC_PROXY_UPSTREAM_TOKENIZE` 時,`count_tokens` 回傳的是**本地 BPE 估算**,而非精確計數。啟用後即可透過 vLLM 風格的 `/tokenize` 取得精確計數。

## 疑難排解

**`UPSTREAM_BASE_URL is required`** —— 必須設定上游端點,例如 `https://openrouter.ai/api`、`https://api.openai.com` 或 `http://localhost:11434`。

**`405 Method Not Allowed` / 上游路徑錯誤** —— 檢查 `UPSTREAM_BASE_URL` 如何解析(見上方形式)。像 `.../chat` 的部分路徑、以及帶 query string/fragment 的 URL 會被拒絕。

**找不到模型** —— 設定 `REASONING_MODEL` / `COMPLETION_MODEL`,或用 `ANTHROPIC_PROXY_MODEL_MAP` 把客戶端模型名稱對映到上游名稱。請確認你對外公告的模型 id 與 map 的 key 一致。

**Gateway/WAF 以 `403` 擋住 Claude Code 的 prompt** —— 用 `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS` / `--system-prompt-ignore` 移除被擋的詞彙。

**負載下間歇性 `502`** —— 確認使用的是最新建置(已含過期連線強化 + 重試)。若 502 持續,代表上游本身回傳了傳輸失敗;以 `--debug` 檢查代理日誌。

## 開發

```bash
cargo test          # 單元測試
cargo clippy        # lint
cargo fmt           # 格式化
```

## 授權

MIT License —— Copyright (c) 2025 m0n0x41d (Ivan Zakutnii)。Fork 由 ExpTech 維護。詳見 [LICENSE](LICENSE)。

## 連結

- [Anthropic API 文件](https://docs.anthropic.com/)
- [OpenRouter 文件](https://openrouter.ai/docs)
- [Rust 文件](https://doc.rust-lang.org/)
