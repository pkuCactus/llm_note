# 投机推理（Speculative Decoding）

## 1. 背景：普通自回归推理为什么慢？

LLM 生成文本是自回归的：

```
p(x₁, x₂, ..., xₜ) = ∏ₜ₌₁ᵀ p(xₜ | x<ₜ)
```

普通解码流程是：

1. 输入 prompt
2. 大模型生成 token 1
3. 把 token 1 加回上下文
4. 大模型生成 token 2
5. 把 token 2 加回上下文
6. 大模型生成 token 3
7. ...

每一步都必须调用一次大模型。

如果要生成 100 个 token，就要进行大约 100 次大模型 decode forward。

而 decode 阶段的特点是：

- 每次只生成 1 个 token
- batch 很小或有效计算密度低
- 要反复读模型权重和 KV Cache
- GPU 很多时候不是算力瓶颈，而是访存/调度瓶颈

所以单 token 解码的吞吐不高。

---

## 2. 投机推理的核心思想

假设有两个模型：

| 模型 | 作用 |
|:-----|:-----|
| 大模型 p | 目标模型，质量高，但慢 |
| 小模型 q | 草稿模型，质量较低，但快 |

我们真正想从大模型分布 p 生成文本。

投机推理让小模型先连续猜多个 token：

```
小模型猜：A B C D
```

然后大模型不是一个一个验证，而是并行计算这些位置上的概率：

```
大模型一次 forward 验证：
token A 是否合理
token B 是否合理
token C 是否合理
token D 是否合理
```

如果前几个 token 被接受，就可以一次推进多个位置。

例如：

```
小模型猜：A B C D
大模型接受：A B C
大模型拒绝：D
```

这一次大模型调用相当于生成了 4 个位置的信息，最终可能产出 3 到 5 个 token，而不是只产出 1 个 token。

---

## 3. 最简单直觉版流程

设当前位置上下文是：

```
The capital of France is
```

小模型先猜 4 个 token：

```
Paris . It is
```

然后大模型一次性看这些候选 token 的每一步概率：

```
p(Paris | The capital of France is)
p(. | The capital of France is Paris)
p(It | The capital of France is Paris .)
p(is | The capital of France is Paris . It)
```

如果大模型也认为这些 token 很合理，那么它们就被接受。

于是原来大模型需要这样：

```
大模型生成 Paris
大模型生成 .
大模型生成 It
大模型生成 is
```

现在变成：

```
小模型快速猜 4 个
大模型一次验证 4 个
```

如果接受率高，就省了很多大模型调用。

---

## 4. 为什么大模型可以"一次验证多个 token"？

这是投机推理最关键的工程点之一。

Transformer 在 prefill 阶段本来就能并行处理一段序列。

小模型给出草稿：

```
y₁, y₂, ..., yₖ
```

大模型可以把这 K 个 token 一起输入，然后通过 causal mask 同时得到每个位置的预测分布：

```
p₁ = p(· | x)
p₂ = p(· | x, y₁)
p₃ = p(· | x, y₁, y₂)
...
pₖ₊₁ = p(· | x, y₁, ..., yₖ)
```

也就是说，一次大模型 forward 可以拿到多个条件分布。

这类似把原来的多个 decode step 合并成一个短 prefill。

---

## 5. 关键问题：怎么保证正确性？

如果只是"小模型猜，大模型同意就接受"，听起来可能会改变大模型的生成分布。

真正的投机采样有一套接受-拒绝机制，保证最终采样结果严格等价于从大模型 p 采样。

下面讲标准 speculative sampling。

---

## 6. 标准投机采样算法

假设：

- 目标模型分布是 p
- 草稿模型分布是 q
- 小模型一次草拟 K 个 token
- 当前位置上下文为 x

### 第一步：小模型生成草稿 token

小模型依次采样：

```
y₁ ~ q(· | x)
y₂ ~ q(· | x, y₁)
...
yₖ ~ q(· | x, y₁, ..., yₖ₋₁)
```

得到草稿序列：

```
y₁, ..., yₖ
```

### 第二步：大模型计算对应概率

大模型一次 forward 计算：

```
p(yᵢ | x, y<ᵢ)
```

同时也能得到这些位置上完整的分布 pᵢ。

小模型在生成草稿时也知道：

```
q(yᵢ | x, y<ᵢ)
```

### 第三步：逐个 token 做接受判断

对于第 i 个草稿 token yᵢ，接受概率是：

```
aᵢ = min(1, p(yᵢ | x, y<ᵢ) / q(yᵢ | x, y<ᵢ))
```

然后抽一个随机数 r ~ U(0,1)。

如果：

```
r ≤ aᵢ
```

则接受 yᵢ。

否则拒绝 yᵢ，停止验证后面的草稿 token。

### 第四步：如果拒绝，如何重新采样？

如果第 i 个 token 被拒绝，不能直接从 p 采样，否则分布会出错。

正确做法是从修正分布中采样：

```
p_res(z) ∝ max(0, pᵢ(z) - qᵢ(z))
```

也就是：

```
p_res(z) = max(0, pᵢ(z) - qᵢ(z)) / Σᵥ max(0, pᵢ(v) - qᵢ(v))
```

这里 v 遍历词表。

然后把采样到的 token 作为当前位置输出，结束本轮。

### 第五步：如果全部接受

如果 K 个草稿 token 都被接受，还可以额外从大模型的下一个分布采样一个 token：

```
z ~ p(· | x, y₁, ..., yₖ)
```

所以一轮最多可以输出 K+1 个 token。

---

## 7. 为什么这个算法保持正确性？

核心是接受-拒绝采样。

我们先看单步情况。

小模型从 q 采了一个 token y。如果接受概率是：

```
min(1, p(y) / q(y))
```

那么某个 token y 通过"小模型采样 + 被接受"产生的概率是：

```
q(y) min(1, p(y) / q(y))
```

这等于：

```
min(q(y), p(y))
```

也就是说，被直接接受的部分对应的是 p 和 q 的重叠区域。

但是目标分布 p 里面还有一部分没有被覆盖，尤其是：

```
p(y) > q(y)
```

的那些 token。

这些剩余概率质量就是：

```
p(y) - min(p(y), q(y)) = max(0, p(y) - q(y))
```

所以如果拒绝了，就从：

```
max(0, p(y) - q(y)) / Σᵥ max(0, p(v) - q(v))
```

这个残差分布里采样，正好补齐 p 的剩余部分。

因此最终 token 的分布就是：

```
min(p(y), q(y)) + max(0, p(y) - q(y)) = p(y)
```

所以单步严格等价于从 p 采样。

多 token 情况下，因为每一步都在对应上下文上保持条件分布正确，所以整个序列分布也与目标模型一致。

---

## 8. 一个离散例子帮助理解

假设词表只有三个 token：

```
A, B, C
```

大模型分布 p：

| token | p |
|:------|:---|
| A | 0.5 |
| B | 0.3 |
| C | 0.2 |

小模型分布 q：

| token | q |
|:------|:---|
| A | 0.4 |
| B | 0.4 |
| C | 0.2 |

小模型先采样。

如果采到 A：

```
a(A) = min(1, 0.5/0.4) = 1
```

A 一定接受。

如果采到 B：

```
a(B) = min(1, 0.3/0.4) = 0.75
```

B 有 75% 接受。

如果采到 C：

```
a(C) = min(1, 0.2/0.2) = 1
```

C 一定接受。

直接接受部分概率是：

| token | 接受概率质量 |
|:------|:-------------|
| A | 0.4 × 1 = 0.4 |
| B | 0.4 × 0.75 = 0.3 |
| C | 0.2 × 1 = 0.2 |

总接受质量是：

```
0.4 + 0.3 + 0.2 = 0.9
```

还缺大模型分布中的：

```
A: 0.1
B: 0
C: 0
```

所以拒绝时残差分布只会采 A。

最终：

| token | 总概率 |
|:------|:-------|
| A | 0.4 + 0.1 = 0.5 |
| B | 0.3 |
| C | 0.2 |

正好等于大模型 p。

---

## 9. 为什么能加速？

投机推理能加速主要靠三个因素。

### 9.1 用小模型廉价地产生多个 token

小模型 q 通常比大模型 p 小很多，例如：

```
draft model: 1B / 3B
target model: 13B / 70B
```

小模型生成 K 个 token 的成本，可能还比大模型生成 1 个 token 低很多。

如果小模型足够快，草稿成本可以忽略一部分。

### 9.2 大模型一次验证多个 token

普通推理：

```
大模型 forward 1 次 -> 生成 1 token
大模型 forward 1 次 -> 生成 1 token
大模型 forward 1 次 -> 生成 1 token
```

投机推理：

```
小模型生成 4 token
大模型 forward 1 次 -> 验证这 4 token，并可能额外生成 1 token
```

如果平均每轮接受 3 个草稿 token，再额外生成 1 个 token，那么一次大模型 forward 可能产出约 4 个 token。

大模型调用次数减少，延迟自然下降。

### 9.3 大模型验证阶段更像 prefill，计算并行度更高

decode 单 token 的矩阵乘法通常 batch 小、序列短，GPU 利用率不一定高。

而一次验证多个 token，相当于让大模型处理一个短序列 chunk：

```
K 个 draft token 一起算
```

这比单 token decode 有更好的并行度。

所以不仅调用次数少了，每次调用的硬件利用率也可能更好。

---

## 10. 加速比由什么决定？

加速效果主要取决于：

- 草稿模型速度
- 草稿模型和目标模型的相似度
- 每轮草稿长度 K
- 接受率
- batch size
- 推理框架 kernel 实现
- KV Cache 管理开销

简单估算一下。

假设：

- 小模型生成 K=4 个 token 的成本相当于大模型 0.3 次 forward
- 大模型验证一次成本约为大模型 1 次 forward
- 平均一轮输出 3 个 token

那么普通生成 3 个 token 需要：

```
大模型 3 次 forward
```

投机推理需要：

```
小模型成本 0.3 + 大模型验证 1 = 1.3
```

加速比约为：

```
3 / 1.3 ≈ 2.3
```

实际常见加速可能是 1.3x 到 3x，特别理想时更高。

---

## 11. 接受率为什么重要？

如果小模型猜得准，大模型经常接受：

```
draft: A B C D
accept: A B C D
```

一轮输出 K+1 个 token，收益很大。

如果小模型猜得差：

```
draft: A B C D
reject: A
```

那么大模型验证一次只输出 1 个修正 token，还额外花了小模型成本，反而可能变慢。

所以投机推理的核心条件是：

**draft model 的分布要和 target model 足够接近，同时 draft model 必须足够便宜。**

---

## 12. 贪心解码下的投机推理

上面讲的是采样场景。那如果不用 sampling，而是 greedy decoding 呢？

普通 greedy 是：

```
xₜ = argmax p(· | x<ₜ)
```

投机 greedy 更简单：

- 小模型生成多个 token
- 大模型并行计算每个位置的 argmax
- 从左到右比较
- 只要小模型 token 等于大模型 argmax，就接受
- 第一个不一致的位置，用大模型 argmax 替换

例如：

```
小模型：A B C D
大模型：A B X ...
```

接受：

```
A B
```

然后输出大模型的：

```
X
```

这样也能保证和大模型 greedy 完全一致。

不过实际开放式生成常常用 top-p、temperature，因此需要前面那套概率修正算法。

---

## 13. KV Cache 在投机推理里怎么处理？

投机推理会带来 KV Cache 管理问题。

小模型生成草稿时会产生自己的 KV Cache：

```
draft KV
```

大模型验证时也会产生目标模型的 KV Cache：

```
target KV
```

如果草稿 token 被接受，对应的大模型 KV 可以保留。

如果某些 token 被拒绝：

```
接受 y₁, y₂
拒绝 y₃
```

那么：

- y₁, y₂ 的 target KV 保留
- y₃ 以及后续草稿 token 的 target KV 丢弃
- 拒绝位置重新采样的 token 需要作为真实 token 进入上下文
- draft model 的 KV 也要同步到真实上下文状态

高性能实现里，KV Cache 的 append、rollback、page 回收非常重要。

---

## 14. 投机推理的几种变体

### 14.1 两模型 speculative decoding

最经典：

```
小 draft model + 大 target model
```

**优点：**

- 正确性清晰
- 加速明显
- 理论成熟

**缺点：**

- 需要额外加载一个小模型
- 小模型和大模型 tokenizer、分布最好匹配
- 显存增加

### 14.2 Self-speculative decoding

不用额外小模型，而是用大模型自身的浅层作为 draft。

例如：

```
前 N 层先猜 token
完整模型再验证
```

**优点：**

- 不需要额外模型
- tokenizer 天然一致
- 显存更省

**缺点：**

- 浅层 draft 质量有限
- 需要模型结构或推理框架支持
- 加速比可能不如好的独立 draft 模型

代表思路包括 LayerSkip、early-exit speculative decoding 等。

### 14.3 Medusa

Medusa 给大模型加多个预测头，让模型一次预测未来多个 token。

形式上：

```
主模型 hidden state
接多个 heads:
  head1 预测下一个 token
  head2 预测下下个 token
  head3 预测下下下个 token
  ...
然后用树状候选验证。
```

**优点：**

- 不需要单独 draft 模型
- 可减少 decode 轮数
- 和 speculative verification 思路类似

**缺点：**

- 通常需要额外训练这些 heads
- 工程实现比普通 speculative decoding 复杂

### 14.4 N-gram / Prompt Lookup Decoding

在很多任务中，模型会复制 prompt 或上下文中的片段。

Prompt lookup 的做法是从已有上下文里找可能的 n-gram 延续，当作草稿 token。

例如上下文里已有：

```
machine learning models are powerful
```

当前又出现：

```
machine learning
```

那可以猜：

```
models are powerful
```

**优点：**

- 不需要小模型
- 对复制型任务非常有效，如摘要、代码、RAG 引用
- 成本极低

**缺点：**

- 适用面有限
- 开放式对话里接受率可能不高

### 14.5 Tree-based speculative decoding

不是只生成一条草稿链，而是生成一棵候选树。

例如：

```
        A
      / | \
     B  C  D
    / \
   E   F
```

大模型一次验证多个候选路径，然后选择可接受路径。

**优点：**

- 提高接受 token 数
- 更充分利用大模型一次 forward 的并行能力

**缺点：**

- 实现复杂
- attention mask、KV 管理更麻烦
- 候选过多会增加验证成本

---

## 15. 投机推理什么时候效果好？

**适合：**

- 大模型很大，小模型很快
- 目标模型和草稿模型分布接近
- 输出比较确定，例如代码、翻译、摘要、格式化文本
- temperature 较低
- batch 不太大，单请求延迟重要
- GPU decode 阶段利用率不高

**不适合或收益小：**

- batch 已经很大，GPU 利用率很高
- 草稿模型太慢
- 草稿模型质量差
- 高 temperature 创造性生成，接受率低
- 每个请求很短，额外开销不划算
- 服务系统 KV 管理不够高效

---

## 16. 为什么 temperature 越高接受率可能越低？

temperature 高时，分布更平：

```
p_T(x) ∝ p(x)^(1/T)
```

当 T 较大，采样更随机。

小模型和大模型即使都"合理"，也更容易采到不同 token，导致接受率下降。

低温、greedy、top-k 较小的场景，小模型更容易猜中大模型偏好的 token，因此投机推理更有效。

---

## 17. 和普通 batching 的关系

投机推理主要优化 **单请求或小 batch 的 decode latency**。

如果在线服务已经有大量并发，continuous batching 能把很多用户请求组成大 batch，使 GPU 利用率很高。

这时投机推理的收益可能下降，因为：

- 大模型单步 decode 已经被 batch 填满
- draft model 也要占资源
- 验证不同请求的草稿长度不一致，调度更复杂

但二者也可以结合，只是工程难度更高。

---

## 18. 一个完整伪代码

```python
while not finished:
    # 1. draft model proposes K tokens
    draft_tokens = []
    ctx = current_context

    for i in range(K):
        q_dist = draft_model(ctx)
        y = sample(q_dist)
        draft_tokens.append((y, q_dist))
        ctx = ctx + [y]

    # 2. target model verifies all draft tokens in one forward
    p_dists = target_model.forward_with_context(
        current_context,
        [y for y, _ in draft_tokens]
    )

    # 3. accept/reject from left to right
    accepted = []

    for i in range(K):
        y, q_dist = draft_tokens[i]
        p_dist = p_dists[i]

        accept_prob = min(1.0, p_dist[y] / q_dist[y])

        if random_uniform() <= accept_prob:
            accepted.append(y)
        else:
            residual = relu(p_dist - q_dist)
            residual = residual / residual.sum()
            z = sample(residual)

            output.extend(accepted)
            output.append(z)
            current_context.extend(accepted + [z])
            break

    else:
        # all draft tokens accepted
        next_dist = p_dists[K]
        z = sample(next_dist)

        output.extend([y for y, _ in draft_tokens])
        output.append(z)
        current_context.extend([y for y, _ in draft_tokens] + [z])
```

实际实现会更复杂，尤其是 KV Cache、top-p、temperature、batching 和数值稳定性。

---

## 19. 总结

投机推理解决的是：

**大模型自回归解码一次只能出一个 token，调用频繁且硬件利用率不高的问题。**

它的做法是：

1. 小模型便宜地生成多个候选 token
2. 大模型一次性并行验证这些候选 token
3. 通过接受-拒绝采样保证输出分布不变
4. 如果接受率高，就减少大模型 decode 次数，从而加速

最关键的两点：

| 问题 | 答案 |
|:-----|:-----|
| 为什么能加速？ | 小模型便宜地产生多个 token，大模型一次验证多个 token，减少大模型调用次数 |
| 为什么保持正确？ | 接受概率 min(1, p/q) 加上残差分布 max(0, p-q)，严格补齐目标模型分布 p |

**一句话总结：**

投机推理不是让小模型替代大模型，而是让小模型提前猜，大模型负责裁判；只要裁判规则正确，最终结果仍然等价于大模型自己生成。

----------