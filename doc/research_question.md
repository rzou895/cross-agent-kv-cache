\# Research Question



\## 1. Project title  	\*\*Cross-Agent KV-Cache Sharing on Statically Quantized NPUs\*\*



本项目研究多个智能体在静态量化的大语言模型和 NPU 环境中，如何安全、高效地共享完全相同 token 前缀所对应的 KV cache。



项目的主要目标是减少多个智能体重复处理相同公共上下文所产生的 prefill 计算和 KV-cache 内存开销，同时控制静态量化 calibration mismatch 带来的准确率损失。



\## 2. What is a KV cache?



自回归大语言模型在处理输入和生成新 token 时，会在每一层 self-attention 中计算 Key 和 Value 张量。



模型在生成后续 token 时仍然需要访问前面 token 的 Key 和 Value。如果每生成一个 token 都重新计算之前所有 token 的 Key 和 Value，会产生大量重复计算。



KV cache 会保存已经处理过的 token 对应的 Key 和 Value，使模型在后续 decode 阶段能够直接复用这些中间结果。



因此，KV cache 不是原始文本的副本，也不是普通的文本摘要。它是由特定模型、特定权重、特定 token 序列和特定位置产生的内部张量。



\## 3. What is static quantization?



大语言模型中的权重、激活值和 KV cache 通常以浮点数表示，例如 FP16。量化会把浮点数近似映射为较低位宽的整数，以减少内存占用、数据传输量和硬件计算成本。



本项目第一版使用 INT8 symmetric quantization：



\[q=\\operatorname{clamp}

\\left(

\\operatorname{round}(x/s),

\-127,

127

\\right)

]



\[

\\hat{x}=s\\cdot q

]



其中：



\* (x) 是原始浮点 KV 值；

\* (q) 是量化后的 INT8 整数；

\* (s) 是量化 scale；

\* (\\hat{x}) 是反量化后的近似值。



静态量化表示 scale 在正式推理前通过 calibration dataset 确定。校准结束后，scale 被固定；推理时不允许根据当前输入重新统计 min/max 或重新估计 scale。



\## 4. What are scale and zero-point?



Scale 决定一个整数单位对应多大的浮点数范围。



例如，当：



\[

s=0.1

]



时，量化整数 (8) 近似表示：



\[

8\\times0.1=0.8

]



Zero-point 决定浮点数零对应哪个整数值。本项目第一版采用 symmetric quantization，因此：



\[

zero\\text{-}point=0

]



也就是说，浮点数零直接映射为整数零，正值和负值使用对称的整数范围。



本项目为每一层、每一个 KV head 的 K 和 V 分别设置固定 scale，可表示为：



\[

s^K\_{l,h}

]



和：



\[

s^V\_{l,h}

]



其中 (l) 表示模型层，(h) 表示 KV head。



\## 5. Why can different calibration scales break cache sharing?



本项目为同一个基础模型生成三套 calibration profile：



\* `model\_general`

\* `model\_code`

\* `model\_qa`



三者使用完全相同的：



\* 模型结构；

\* 模型权重；

\* tokenizer；

\* token 输入；

\* attention 配置。



它们唯一的区别是 KV cache 的静态量化 scale 分别由不同校准数据集产生。



假设某个真实 KV 值为：



\[

x=0.8

]



发送方 Agent A 使用：



\[

s\_A=0.1

]



因此保存：



\[

q\_A=8

]



接收方 Agent B 使用：



\[

s\_B=0.2

]



如果 B 自己量化同一个值，它应该保存：



\[

q\_B=4

]



如果直接把发送方的整数 (8) 交给 B，而 B 使用自己的 scale 解码，则 B 会得到：



\[

8\\times0.2=1.6

]



而正确值应接近：



\[

0.8

]



因此，即使两个 Agent 使用同一个模型和同一个输入前缀，不同 calibration scale 仍然可能使相同的 INT8 数字具有不同的浮点含义。



这就是本项目研究的 calibration mismatch。



\## 6. Exact token-prefix sharing constraint



第一版只允许共享从输入位置零开始完全一致的 token-level prefix。



例如：



```text

Agent A:

\[共同 system prompt]\[共同文档]\[问题 A]



Agent B:

\[共同 system prompt]\[共同文档]\[问题 B]



Agent C:

\[共同 system prompt]\[共同文档]\[问题 C]

```



可以共享的部分只有：



```text

\[共同 system prompt]\[共同文档]

```



不同问题开始后，各 Agent 必须分别继续计算自己的 KV cache。



共享长度为 (p) 的公共前缀时，必须满足：



\[

input\_ids\_A\[0:p]=input\_ids\_B\[0:p]

]



还必须保证：



\* 使用相同模型结构和权重；

\* 使用相同 tokenizer；

\* token ID 和 token 顺序完全相同；

\* position ID 或 cache position 完全相同；

\* attention mask 语义相同；

\* RoPE 和其他位置编码配置相同。



不能仅仅因为两段文本内容相似，或同一篇文档在两个 prompt 中都出现，就共享其 KV cache。



如果相同文本前面的 token 不同，或者相同文本出现在不同位置，它产生的隐藏状态和 KV cache 通常也不同。



因此，第一版不研究：



\* 位于不同位置的相同文本共享；

\* 语义相似文本共享；

\* 非连续文本块共享；

\* 跨模型 KV cache sharing；

\* 不同模型权重之间的 KV cache sharing。



\## 7. Primary research question



本项目的主要研究问题是：



> 当多个智能体使用相同的大语言模型、相同权重和完全相同的公共 token 前缀，但使用由不同校准数据分布产生的静态 KV-cache quantization scales 时，如何在维持任务准确率的同时共享公共前缀的 KV cache，并减少重复 prefill 和缓存内存开销？



\## 8. Secondary research questions



1\. 不同 calibration profile 之间的 scale mismatch 会在多大程度上增加 KV reconstruction error、token KL divergence 和 perplexity？



2\. 将发送方 INT8 cache 直接交给接收方并按接收方 scale 解码，会造成多大的任务性能下降？



3\. 运行时 dynamic requantization 能恢复多少准确率，其转换开销是多少？



4\. 预先定义的 canonical shared scale 是否已经足以解决主要问题？



5\. 固定旋转表示是否能够在 canonical shared scale 之上进一步降低量化误差？



6\. divergence check 和 fork 机制能否在限制准确率损失的同时保留大部分内存和计算收益？



\## 9. Methods to compare



本项目第一版比较以下五种方法：



1\. `private\_static`：每个 Agent 独立执行 prefill，并使用自己的静态量化 scale。



2\. `naive\_raw\_share`：直接共享发送方 INT8 cache，但接收方错误地使用自己的 scale 解码。



3\. `dynamic\_requant`：先使用发送方 scale 反量化，再按照接收方 scale 重新量化。



4\. `canonical\_scale`：所有可共享 KV block 使用预先离线确定的统一共享 scale。



5\. `qxcache\_rotation\_fork`：使用固定旋转和共享表示，并通过 divergence check 判断是否需要退出共享路径。



`canonical\_scale` 是必须保留的强基线。只有当旋转方法显著优于简单统一 scale 时，才能说明旋转带来了独立价值。



\## 10. First-version scope



第一版统一采用：



\* INT8 KV cache；

\* symmetric quantization；

\* zero-point 固定为 0；

\* K 和 V 使用不同 scale；

\* 每层、每个 KV head 使用固定 tensor-wise scale；

\* scale 仅在 calibration 阶段生成；

\* evaluation 和 inference 阶段不重新统计 min/max；

\* 只共享完全相同的连续 token prefix。



第一版不同时研究所有 bit width、asymmetric quantization、动态量化、跨模型共享和任意位置文本复用。



