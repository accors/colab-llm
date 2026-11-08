# colab-llm

在 Google Colab 的 GPU 上用 llama.cpp 跑 GGUF 大模型。全部流程通过 [Colab CLI](https://github.com/googlecolab/google-colab-cli) 在本地（Linux / macOS）驱动。模型服务只监听 Colab 那台机器的 `127.0.0.1`，通过 `ssh -L` 映射成本地的 `http://127.0.0.1:8080/v1`（OpenAI 兼容），不暴露到公网。

fork 自 [crhywun/colab-llm](https://github.com/crhywun/colab-llm)。原版用 notebook 加 Cloudflare 隧道，这里改成了 Colab CLI 加 ssh 端口转发。默认配置跑的是 **Qwen3.8-Flash-Next**（125B MoE，激活 6B），GPU 是 **A100 80GB**。

```
本机 agent ──▶ 127.0.0.1:8080 ──ssh -L（colab ssh --proxy-mode）──▶ Colab A100: llama-server 127.0.0.1:8000
                    ▲                                                      ▲
         colab-llm supervisor（后台）                       watchdog（kernel 内常驻，空闲检测）
```

## 文件

| 文件 | 运行位置 | 作用 |
| --- | --- | --- |
| `bin/colab-llm` | 本机 | 唯一入口：`init / prep / up / status / logs / restart / down / chat / ssh / usage` |
| `config.env.example` | 本机 | 配置模板，`init` 时复制成 `config.env` |
| `remote/prep.py` | Colab T4 会话 | 编译 llama-server 存到 HF 私有仓库（或 Drive）；`--models` 时也把模型存到 Drive |
| `remote/serve.py` | Colab GPU 会话 | 检查显存，从 HF 下载模型（或从 Drive 拷贝），启动 llama-server |
| `remote/watchdog.py` | Colab GPU 会话（kernel 内） | 维持会话存活；空闲、超时或服务挂掉时退出 |
| `remote/common.py` | — | 公共函数，发送前拼接到上面三个脚本前面 |
| `chat.py` | 本机 | 命令行对话和测速，只用标准库 |

## 准备

```bash
uv tool install google-colab-cli      # 或 pip install google-colab-cli
colab usage                           # 第一次运行会走 OAuth 登录；显示余额说明登录成功
ssh-keygen -t ed25519                 # 没有 ed25519 密钥时需要生成（Colab 不接受 RSA）

git clone <你的 fork> && cd colab-llm
ln -s "$PWD/bin/colab-llm" ~/.local/bin/colab-llm
colab-llm init                        # 生成 config.env、API key、ssh 配置，并检查登录
```

llama-server 的编译结果存在一个 **HF 私有模型仓库** 里（`BIN_STORE=hf`），这样整个流程都不用挂载 Google Drive，也就不会每台 VM 都要授权一次：

1. 在 Hugging Face 上新建一个私有模型仓库，例如 `accors/colab-llm-bin`，在 `config.env` 里设置 `HF_BIN_REPO=accors/colab-llm-bin`。
2. 新建一个 fine-grained token，只给这个仓库写权限，再加上公开仓库的读权限，然后保存到本机：
   ```bash
   (umask 077; read -rsp 'HF token: ' t; echo; printf '%s\n' "$t" > ~/.config/colab-llm/hf_token)
   ```
   token 通过 ssh 传到 VM，不经过 `colab exec`，所以不会出现在 CLI 的执行记录里。下载模型时也会用这个 token，按你自己账号的配额计算，不会因为和别人共用 Colab 的出口 IP 而被限流（429）。

确保 Colab CLI 登录的 Google 账号就是有计算单元的那个账号。

## 使用

```bash
colab-llm prep --gpu T4   # 只需一次：在 T4 上编译 llama-server 存到 HF_BIN_REPO（约 15 分钟，约 0.3 CU）
colab-llm up          # 开 A100 80G，加载模型，启动后台 supervisor
colab-llm chat "用三句话介绍你自己"
colab-llm status      # 会话、空闲计时、token 数、CU 余额
colab-llm down        # 用完立刻释放 GPU
```

agent 里这样配：Base URL 填 `http://127.0.0.1:8080/v1`，API key 填 `~/.config/colab-llm/api_key` 的内容，模型名填 `qwen3.8-flash-next`。这三项也都写在 `~/.config/colab-llm/client.env` 里。

默认配置（`SOURCE=hf`、`BIN_STORE=hf`）不会用到 Drive。只有设置了 `SOURCE=drive` / `BIN_STORE=drive`，或者使用 `prep --models` / `--from-drive` 时才会挂载 Drive，CLI 会打印一个 Google 授权链接，在浏览器里点同意即可。

## 费用控制

A100 80G 约 **7.52 CU/h**，Colab Pro 每月只有 100 CU，大约 13 小时。下面几层机制用来避免 GPU 空转：

- **空闲自动关闭**：watchdog 每 30 秒读一次 llama-server 的 `/metrics`。连续 `IDLE_MINUTES`（默认 15 分钟）没有处理任何 token，supervisor 就会执行 `colab stop`。
- **单次时长上限**：`MAX_HOURS`（默认 8 小时）。
- **失败即关**：`up` 中途失败（比如分到的是 40G 卡、加载出错）会立即关闭会话，除非加了 `--keep`。
- **显存检查**：显存小于 `MIN_VRAM_GIB`（默认 75）时直接拒绝启动。
- **本机断开**：如果 supervisor 本身挂了，watchdog 照样会在空闲后退出，kernel 进入空闲，由 Colab 后台按自己的空闲策略回收。这种情况下会多烧一段时间，具体多久由 Colab 决定。
- 可以配置 `NOTIFY_CMD`，在会话被自动关闭时发通知（比如 ntfy）。

## 为什么这样设计

- **保活靠 kernel 在执行代码**：Colab CLI 在 2026-09 删掉了保活守护进程，会话是否存活由后台根据 kernel 是否在执行来判断。所以 watchdog 是一个在 kernel 里一直运行的 `colab exec --timeout 0`。代价是它运行期间 kernel 被占住，`restart` 会先让它退出，再重新执行 serve。
- **不会误开会话**：`colab ssh --proxy-mode -s NAME` 在会话不存在时会自动新建一个 CPU 会话。所以每次 ssh 之前都先确认会话存在；supervisor 如果发现同名会话变成了非 GPU 会话，也会把它关掉。
- **API key 不经过 `colab exec`**：exec 会把发送的代码（包括 `--env` 生成的前置代码）记录到 `~/.config/colab-cli/history`，所以 key 改走 ssh 传过去，llama-server 用 `--api-key-file` 读取。
- **共用一条 SSH 连接**：每个运行时的 SSH 并发连接数有上限，所以用 ControlMaster，隧道和其他命令共用同一条连接。
- **n-gram 表不占显存**：使用 `-lm mmap --lazy-mode on`，28.8 GB 的 n-gram 分片保持 mmap 在本地盘上按需读取，显存里只放约 55 GB 的权重分片。

## 换模型或量化

修改 `config.env` 里的 `HF_REPO` / `QUANT`，再运行 `colab-llm restart`（或者 `down` 之后再 `up`）。默认 `SOURCE=hf`，所以不用重新跑 prep。

- 去审查版：`HF_REPO=SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`。它是把替换后的张量写进同一份量化文件，原版 llama.cpp 能直接加载，但**质量和工具调用都没有公开测试**，用之前自己回归测一下。
- alesha-pro 的 control vector 方案在 llama.cpp 里**不会生效**：上游 `qwen4exp` 的实现没有调用 `build_cvec`，只有 Strata 能用。
- 想用 MTP 投机解码，可以设置 `MTP_REPO` / `MTP_FILE`（见配置注释）。速度收益没有实测过。

## 首次使用时需要确认的事项

下面这些在写代码时只能根据文档和源码推断，第一次真实运行时请留意：

1. `--gpu A100 --high-mem` 分到的是不是 80G 卡。`up` 会打印显存大小，不够会直接停掉会话。
2. Colab 那边的 sshd 是否允许 `ssh -L` 端口转发。如果不允许，`up` 最后会提示本地端口不通，需要查看 `colab-llm logs supervisor`。
3. CPU 会话里没有 `nvcc`（已实测）。所以要编译就运行 `colab-llm prep --gpu T4`：T4 约 1.2 CU/h，交叉编译 sm80 大约 15 分钟，约 0.3 CU；如果拖到第一次 `up` 时在 A100 上编译，要 1.25–2.5 CU。
4. 实测：HF 下载约 311 MB/s，写 Drive 约 121–148 MB/s。所以建议 `SOURCE=hf`，每次开机约 4.5 分钟、0.56 CU；从 Drive 读预计约 10 分钟、1.25 CU。

## 命令速查

```text
colab-llm prep --gpu T4           在 T4 上编译 llama-server（CPU 运行时没有 nvcc）
colab-llm prep --gpu T4 --rebuild 强制重新编译（更新 llama.cpp）
colab-llm prep --from-drive       把旧版本存在 Drive 上的编译结果迁移到 HF_BIN_REPO（CPU 会话）
colab-llm up [--keep]             --keep 失败时不关闭会话，方便排查
colab-llm logs server -f          远端 llama-server 日志
colab-llm logs watchdog|supervisor|serve|prep
colab-llm ssh [CMD...]            进入 GPU 会话的 shell，或执行一条命令
colab-llm usage                   计算单元余额
```

## 安全

- 远端的 llama-server 只监听 `127.0.0.1`，本地转发也只绑定在 `127.0.0.1`，同时有 API key 保护。
- `config.env` 已加入 `.gitignore`。API key、HF token 和 ssh 配置都放在 `~/.config/colab-llm/`（权限 700）。
- VM 上的 `api_key` / `hf_token` 文件权限是 600，VM 释放后随之销毁。
