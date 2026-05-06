# llm_note

LLM 学习笔记仓库，记录大语言模型相关的技术知识点。

## 目录结构

```
llm_note/
├── inference/        # 推理优化
├── compression/      # 模型压缩
├── training/         # 训练相关
└── application/      # 应用场景
```

## 内容概览

### 推理优化 (inference/)

- [KV Cache 优化](inference/kv-optimization.md) - KV Cache 显存优化技术详解，包含 MQA/GQA/MLA、量化、分页管理、滑动窗口等
- [投机推理](inference/speculative-decoding.md) - Speculative Decoding 原理、正确性证明及工程实现

### 模型压缩 (compression/)

- [量化](compression/quantization.md) - 待完成
- [剪枝](compression/prune.md) - 待完成

### 训练相关 (training/)

待补充

### 应用场景 (application/)

待补充

## 文档风格

- 语言：中文
- 格式：Markdown（表格、公式、代码示例）
- 特点：技术深度解析，包含理论基础和工程实践

## 仓库特点

这是一个知识库，不包含可执行代码。每个文档都是独立的技术专题，适合系统性学习。