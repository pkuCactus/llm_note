# KV Cache Optimization

## KV Cache优化分类

| 类别        | 方法                                 | 主要解决问题        |
| :----------- | :------------------------------------ | :--------------- |
| 架构优化      | MQA、GQA、MLA                        | 从源头减少KV的数量    |
| 参数量化      | INT8、FP8、INT4                      | 减少单个KV元素的大小   |
| 上下文裁剪     | Sliding Window、Attention Sink      | 限制KV长度        |
| 动态优化      | Token Eviction、KV Merging          | 删除/压缩不重要的历史KV |
| Kernel 优化 | Flash Attention、Flash Decoding | 提高KV访问效率      |
| KV复用      | Prefix Cache、 Prompt Cache         | 减少重复prefill   |
| 内存管理      | PagedAttention、Continuous Batching | 提升显存利用率       |
| 分层存储      | CPU、NVMe Offloading                | 缓解显存不足的问题     |

---

## 1. KV Cache 为什么占显存？

Transformer 自回归推理时，每生成一个 token，都要把该层 attention 的 Key 和 Value 保存下来，之后生成新 token 时复用。

对于一个模型，KV Cache 大小大致是：

```
KV Cache = 2 × L × B × S × H_kv × D × bytes
```

其中：

| 符号 | 含义 |
|:-----|:-----|
| L | 层数 |
| B | batch size / 并发数 |
| S | sequence length |
| H_kv | KV heads 数量 |
| D | 每个 head 的维度 |
| 2 | K 和 V 两份 |
| bytes | 每个元素字节数，例如 FP16 是 2 bytes |

所以长上下文时，KV Cache 会随 S 线性增长。

---

## 2. 减少 KV Cache 数量：MQA / GQA / MLA

### 2.1 MHA：普通多头注意力

普通 Multi-Head Attention 中，Query、Key、Value 都有相同数量的 head。

例如：

```
Q heads = 32
K heads = 32
V heads = 32
```

KV Cache 要保存 32 个 K head 和 32 个 V head。

### 2.2 MQA：Multi-Query Attention

MQA 的做法是：

```
Q heads = 32
K heads = 1
V heads = 1
```

所有 Query head 共享同一组 K/V。

**优点：**
- KV Cache 大幅减少
- 解码阶段访存压力降低
- 推理速度更快

**缺点：**
- 表达能力可能下降
- 对模型训练方式有要求，不能随便把已有 MHA 模型改成 MQA

典型应用：早期很多推理友好模型会采用 MQA。

### 2.3 GQA：Grouped-Query Attention

GQA 是 MHA 和 MQA 的折中。

例如：

```
Q heads = 32
K/V heads = 8
```

每 4 个 Query head 共享一个 K/V head。

**优点：**
- KV Cache 比 MHA 少很多
- 效果通常比 MQA 更稳
- 现在很多 LLM 默认采用 GQA

例如 LLaMA 2 70B、LLaMA 3、Qwen、Mistral 等都大量使用 GQA。

KV Cache 减少比例大致是：

```
H_kv / H_q
```

如果从 32 个 KV heads 减到 8 个，就是减少到原来的 1/4。

### 2.4 MLA：Multi-head Latent Attention

MLA 是 DeepSeek 系列里非常重要的 KV Cache 优化方法。

普通注意力保存完整的 K/V，而 MLA 不是直接缓存完整 K/V，而是缓存更低维的 latent 表示。推理时再从 latent 表示恢复出需要的 K/V 相关信息。

直觉上：

```
普通方式：缓存完整 K/V
MLA：缓存压缩后的 latent KV
```

**优点：**
- KV Cache 显著降低
- 长上下文推理更省显存
- 对大规模 MoE/长上下文模型很有价值

**缺点：**
- 架构级改动
- 需要从训练阶段就设计进去
- 不是普通 MHA 模型可以简单后处理得到的

---

## 3. 量化 KV Cache

这是最常见的工程优化之一。

### 3.1 FP16/BF16 KV Cache

默认通常是：

```
K/V: FP16 或 BF16
每个元素 2 bytes。
```

如果上下文很长，比如 32K、128K，KV Cache 会非常大。

### 3.2 INT8 KV Cache

把 KV Cache 从 FP16 量化到 INT8：

```
2 bytes -> 1 byte
理论上显存减半。
```

**优点：**
- 显存减少约 50%
- 对长上下文、多并发非常有用
- 硬件支持较好

**缺点：**
- 需要 scale/zero point
- 注意力分数对 K 的精度比较敏感
- 可能带来困惑度或生成质量下降

常见策略：
- per-token quantization
- per-channel quantization
- per-head quantization
- K/V 分别使用不同量化策略

### 3.3 INT4 / FP8 KV Cache

更激进的做法是 INT4 或 FP8。

```
FP16: 2 bytes
FP8:  1 byte
INT4: 0.5 byte
```

FP8 在新 GPU 上比较有吸引力，比如 H100、H200、B200 等。

INT4 压缩率更高，但难度更大，因为 attention 对 KV 精度很敏感。

一般来说：

| KV 精度 | 显存节省 | 难度 | 质量风险 |
|:--------|:---------|:-----|:---------|
| FP16/BF16 | 无 | 低 | 低 |
| FP8 | 约 50% | 中 | 中低 |
| INT8 | 约 50% | 中 | 中 |
| INT4 | 约 75% | 高 | 较高 |

---

## 4. KV Cache 分页管理：PagedAttention

PagedAttention 是 vLLM 里的核心技术。

传统 KV Cache 管理常常需要为每个请求预留连续显存。问题是：

- 不同请求长度不同
- 有的请求提前结束
- 预分配会浪费
- 连续内存难以管理
- 多轮对话下碎片严重

PagedAttention 借鉴操作系统的分页思想，把 KV Cache 切成固定大小的 block/page。

例如：

```
Request A: page 1 -> page 5 -> page 9
Request B: page 2 -> page 3
Request C: page 4 -> page 7 -> page 8 -> page 10
```

逻辑上每个请求的 KV 是连续的，但物理显存上可以不连续。

**优点：**

- 显著降低显存碎片
- 支持更高并发
- 动态分配和释放方便
- 适合 continuous batching

这类优化并不减少单个 token 的 KV 大小，但能提升显存利用率。

---

## 5. Sliding Window Attention

Sliding Window Attention 的思路是：只保留最近一段窗口内的 KV Cache。

例如窗口大小是 4096，那么生成第 10000 个 token 时，只关注最近 4096 个 token：

```
只看 tokens [5905, 10000]
丢弃更早的 KV
```

显存从随总长度增长变成随窗口大小增长：

```
O(S) → O(W)
```

其中 W 是窗口长度。

**优点：**

- KV Cache 有上限
- 长文本生成显存稳定
- 速度更快

**缺点：**

- 模型无法直接关注窗口外的信息
- 长程依赖能力受限
- 需要模型结构或训练时支持

典型模型：Mistral 使用 sliding window attention。

---

## 6. Attention Sink / StreamingLLM

纯 sliding window 有个问题：LLM 往往很依赖最开始的几个 token，例如 system prompt、格式指令等。

Attention Sink 的观察是：

保留开头少量 token 的 KV，再保留最近窗口的 KV，可以让流式长文本生成更稳定。

形式上：

```
保留：开头 sink tokens + 最近 window tokens
丢弃：中间较旧 tokens
```

例如：

```
[前 4 个 token] + [最近 4096 个 token]
```

**优点：**

- 比单纯 sliding window 稳定
- 支持无限长度流式生成
- KV Cache 规模固定

**缺点：**

- 中间历史信息仍然丢失
- 对需要精确长程回忆的任务不理想

---

## 7. Prefix Cache / Prompt Cache

很多服务场景里，不同请求会共享相同前缀，比如：

- System prompt
- 工具说明
- RAG 模板
- 角色设定
- 长文档前缀

如果每次请求都重新计算这些 prefix 的 KV，就很浪费。

Prefix Cache 的做法是：

对相同 prompt 前缀预先计算 KV Cache，并在后续请求中复用。

**优点：**

- 减少 prefill 计算
- 降低首 token 延迟 TTFT
- 对 Agent、RAG、多轮对话很有用

**挑战：**

- 如何判断 prefix 相同
- tokenizer 后的 token 序列必须匹配
- cache 失效和淘汰策略
- 多租户隔离
- 显存/CPU 内存之间的缓存管理

vLLM、TensorRT-LLM、SGLang 等框架都支持类似功能。

---

## 8. KV Cache Offloading

如果 GPU 显存不够，可以把部分 KV Cache 放到：

- CPU 内存
- NVMe SSD
- 另一张 GPU
- 分层存储系统

常见策略：

```
热 KV：GPU
温 KV：CPU
冷 KV：SSD
```

**优点：**

- 支持更长上下文
- 支持更多并发
- 减少 GPU 显存压力

**缺点：**

- PCIe/NVLink 传输有延迟
- 如果频繁访问被 offload 的 KV，会明显变慢
- 需要好的调度策略

**适合：**

- 超长上下文
- 多轮对话历史
- 低 QPS 但长序列任务
- 显存紧张部署

---

## 9. KV Cache 压缩 / Eviction

这类方法尝试判断哪些历史 token 不重要，然后丢弃或压缩它们。

### 9.1 Token eviction

根据注意力分数、最近性、重要性等，丢掉一部分 KV。

例如：

- 保留最近 token
- 保留被高频关注的 token
- 保留特殊 token
- 删除低 attention token

代表思路包括 H2O、Scissorhands 等。

**优点：**

- 可以动态减少 KV
- 比固定窗口更灵活

**缺点：**

- 判断错误会损害生成质量
- attention 分数不一定等于长期重要性
- 实现复杂

### 9.2 KV merging / clustering

不是直接丢弃 token，而是把多个历史 token 的 KV 合并。

类似：

```
token 10, 11, 12 的 KV -> 一个压缩 KV
```

**优点：**

- 保留部分历史信息
- 比直接丢弃温和

**缺点：**

- 算法更复杂
- 可能改变 attention 分布
- 部署框架支持较少

---

## 10. FlashAttention / FlashDecoding

严格来说，FlashAttention 主要优化 attention 计算的 IO，而不是直接减少 KV Cache 容量。

但它和 KV Cache 优化密切相关，因为解码阶段瓶颈常常是：

```
读取历史 K/V -> 计算 attention -> 得到输出
```

FlashAttention / FlashDecoding 会通过更好的 kernel 设计减少 HBM 读写，提高吞吐。

**优点：**

- 降低显存读写开销
- 提高 attention 速度
- 对长上下文非常重要

**缺点：**

- 不直接减少 KV Cache 存储量
- 对硬件和 kernel 实现依赖较强

---

## 11. Speculative Decoding 和 KV Cache

投机解码本身主要是减少大模型前向次数，但它也涉及 KV Cache 管理。

过程大致是：

- 小模型 draft 多个 token
- 大模型一次验证多个 token
- 接受的 token 写入 KV Cache
- 拒绝的 token 需要回滚或丢弃 KV

优化点包括：

- draft KV 和 target KV 分开管理
- rejected tokens 的 KV 快速回收
- tree-based speculative decoding 的 KV 共享

这类方法不是为了压缩 KV，而是为了更高解码吞吐。

---

## 12. Continuous Batching 中的 KV Cache 管理

在线服务中，请求不断进入和结束。Continuous batching 会把不同阶段的请求动态组成 batch。

问题是每个请求的 KV Cache 长度不同，需要高效管理。

常见优化：

- block/page 分配
- 空闲 block 回收
- prefix cache 复用
- request 级别调度
- decode/prefill 分离
- chunked prefill

这些通常和 PagedAttention 一起使用。

---

## 13. Chunked Prefill

长 prompt 的 prefill 会一次性消耗大量显存和计算，影响其他 decode 请求。

Chunked Prefill 把长 prompt 分块处理：

```
一个 64K prompt
拆成 8 个 8K chunk
逐块 prefill
```

**优点：**

- 降低峰值显存
- 避免长 prompt 阻塞短请求
- 更适合在线服务调度

**缺点：**

- 调度复杂
- TTFT 可能受策略影响

---

## 14. 分布式 KV Cache

在多 GPU 推理中，KV Cache 可以按不同维度切分：

### 14.1 Tensor Parallel 下的 KV

每张 GPU 保存自己 attention head 对应的 KV。

**优点：**

- 自然匹配多头切分
- 减少单卡 KV 压力

**缺点：**

- 需要通信聚合 attention 输出

### 14.2 Context Parallel / Sequence Parallel

把长序列维度切到多张 GPU 上：

```
GPU0: tokens 0-8191
GPU1: tokens 8192-16383
GPU2: tokens 16384-24575
...
```

适合超长上下文。

**优点：**

- 单卡 KV Cache 下降
- 支持更长上下文

**缺点：**

- attention 需要跨卡通信
- 实现复杂
- 延迟可能上升

---

## 15. 实际工程中常见组合

不同场景会组合多种方法。

### 场景 1：普通聊天服务

常用：

- GQA 模型
- PagedAttention
- Prefix Cache
- Continuous Batching
- FP16/BF16 KV

**目标：**高吞吐、稳定质量。

### 场景 2：长上下文 RAG

常用：

- GQA / MLA
- KV Cache INT8 或 FP8
- Prefix Cache
- Chunked Prefill
- PagedAttention

**目标：**降低 TTFT 和显存占用。

### 场景 3：无限流式生成

常用：

- Sliding Window
- Attention Sink
- KV eviction
- KV quantization

**目标：**固定 KV Cache 上限。

### 场景 4：显存非常紧张

常用：

- INT8/INT4 KV Cache
- CPU offloading
- Sliding Window
- Prefix Cache
- 小 batch 调度

**目标：**牺牲部分速度，换取可运行。

---

## 16. 总结

KV Cache 优化大致可以分成几类：

| 类别 | 方法 | 主要解决 |
|:-----|:-----|:---------|
| 架构优化 | MQA、GQA、MLA | 从源头减少 KV 数量 |
| 精度优化 | INT8、FP8、INT4 KV | 减少单个 KV 元素大小 |
| 内存管理 | PagedAttention、Continuous Batching | 提升显存利用率 |
| 上下文裁剪 | Sliding Window、Attention Sink | 限制 KV 长度 |
| 复用优化 | Prefix Cache、Prompt Cache | 减少重复 prefill |
| 分层存储 | CPU/NVMe Offloading | 缓解 GPU 显存不足 |
| 动态压缩 | Token eviction、KV merging | 删除/压缩不重要历史 |
| Kernel 优化 | FlashAttention、FlashDecoding | 提高 KV 访问效率 |

**一句话概括：**

KV Cache 优化要么减少"存多少"，要么减少"每个元素多大"，要么提高"怎么存、怎么读、怎么复用"的效率。

----------
