# Larme Relay — 免费模型中转站

OpenAI 兼容的免费模型网关，跑在本机 CPU 上（llama.cpp + Q4 量化），按需加载、闲置自动卸载。

## 地址

```
https://platform.we-strikes.top/relay/v1/chat/completions
https://platform.we-strikes.top/relay/v1/models
https://platform.we-strikes.top/relay/            # 说明页
```

不带 Key，限流每 IP 每分钟 60 次，支持 `stream: true`。

## 模型

| 模型名 | 底座 | 文件 | 上下文 | 对话模板 |
|---|---|---|---|---|
| `qwen2.5-0.5b` | Qwen2.5-0.5B-Instruct | 469 MB | 2048 | chatml |
| `qwen2.5-1.5b` | Qwen2.5-1.5B-Instruct | 1.1 GB | 2048 | chatml |
| `qwen2.5-3b` | Qwen2.5-3B-Instruct | 1.8 GB | 2048 | chatml |
| `qwen2.5-coder-1.5b` | Qwen2.5-Coder-1.5B-Instruct | 940 MB | 2048 | chatml |
| `qwen3-1.7b` | Qwen3-1.7B | 1.2 GB | 2048 | chatml |
| `deepseek-r1-1.5b` | DeepSeek-R1-Distill-Qwen-1.5B | 1.1 GB | 2048 | chatml |
| `glm-edge-1.5b` | GLM-Edge-1.5B-Chat | 935 MB | 4096 | chatml |
| `gemma-3-1b` | Gemma-3-1B-it | 768 MB | 4096 | gemma |
| `gemma-3-4b` | Gemma-3-4B-it | 2.4 GB | 2048 | gemma |
| `gpt-oss-4b` | gpt-oss-4B（社区重制，非 OpenAI 官方） | 3.1 GB | 2048 | 模型自带 |
| `qwen3-0.6b` | Qwen3-0.6B | 462 MB | 4096 | 跳过思考 |
| `qwen3-4b` | Qwen3-4B | 2.4 GB | 2048 | 跳过思考 |
| `glm-edge-4b` | GLM-Edge-4B-Chat | 2.5 GB | 2048 | chatml |

别名：
`qwen` / `qwen2.5` / `tongyi` → `qwen2.5-3b`；`default` / `larme` → `qwen2.5-1.5b`；
`coder` / `qwen-coder` → `qwen2.5-coder-1.5b`；`qwen3` → `qwen3-1.7b`；
`deepseek` / `r1` / `ds` → `deepseek-r1-1.5b`；`glm` / `zhipu` → `glm-edge-1.5b`；
`larme-debug` → `qwen2.5-0.5b`。

## 调用

```bash
curl https://platform.we-strikes.top/relay/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-1.5b","messages":[{"role":"user","content":"你好"}]}'
```

Python：

```python
from openai import OpenAI
client = OpenAI(base_url="https://platform.we-strikes.top/relay/v1", api_key="free")
print(client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=[{"role": "user", "content": "你好"}],
).choices[0].message.content)
```

## 架构

```
公网 → platform.we-strikes.top/relay/*  (server.py 反代)
     → 127.0.0.1:13320                 (relay.py 网关，按 model 路由)
     → 127.0.0.1:13400-13409           (model_server.py，每个模型一个进程)
```

端口分配见 `models.json`：13400 每个模型递增，`gpt-oss-4b` 13407、`gemma-3-1b` 13408、`gemma-3-4b` 13409。

## 服务管理

```bash
systemctl --user status larme-relay
systemctl --user restart larme-relay
systemctl --user start larme-model@qwen2.5-3b     # 手动拉起单个模型
```

网关 `larme-relay.service` 常驻；模型走模板 `larme-model@.service`，由网关按需 `start`，闲置 20 分钟 `stop`。

## 本机能力边界（实测）

| 项 | 实测 |
|---|---|
| 单进程内存上限 | 约 3.85 GB（超过即 OOM） |
| 已跑通的4B | Gemma-3-4B，RSS 峰值 2.4 GB |
| 7B Q4 | 需约 5 GB，跑不动 |
| 0.5B 速度 | 空闲时约 20–30 token/s |
| 1.5B 速度 | 空闲时约 8–12 token/s |

## 增加模型

1. 下载 GGUF 到 `/home/agentuser/models/gguf/`
2. 在 `models.json` 加一条（file / port / ctx / label，必要时 `chat_format`）
3. `systemctl --user daemon-reload && systemctl --user restart larme-relay`

不需要单独写 systemd 单元——模板 `larme-model@.service` 用模型名当参数。

## 推理等级（v 标签）

模型名后面加 `-ultra` / `-max` / `-high` 即为同一份权重的三档推理预算：

| 等级 | 标签 | max_tokens 下限 | 温度 |
|---|---|---|---|
| **ultra** | Ultra 最厉害 | 2048 | 0.7 |
| **max** | Max 高智商 | 1024 | 0.6 |
| **high** | High 中智商 | 512 | 0.5 |

```bash
curl .../relay/v1/chat/completions -d '{"model":"qwen2.5-3b-ultra", ...}'
```

不带后缀＝默认档。等级只影响生成预算与采样温度，不改变权重；`/v1/models` 的 `tiers` 字段会列出全部三档。

## 对话模板差异（踩过的坑）

| 模型 | 处理 |
|---|---|
| 千问 2.5 / DeepSeek-Distill / GLM-Edge | `chatml` |
| Gemma-3 | `chat_format="gemma"`，用 chatml 会答非所问 |
| gpt-oss | `chat_format=None`，用 GGUF 自带模板 |
| Qwen3 | 会先输出大段 `<think>`，CPU 上把 token 预算吃光；服务端手拼 `<think>\n\n</think>` 强制跳过思考 |

输出侧还有一道 `去思考()`，任何模型漏出的 `<think>…</think>` 都会被剥掉，前端看不到。

## 已排除

按所有者要求只保留千问、DeepSeek、智谱 GLM 三家，外加 Gemma 与社区 gpt-oss。
Llama / Gemma 之外的 Phi、InternLM、MiniCPM、混元、文心等不进入中转站。
