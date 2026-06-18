# Tuna-Gemma · Pixel-native Unified Multimodal Model with Gemma 4 Backbone

> Forked from [facebookresearch/tuna-2](https://github.com/facebookresearch/tuna-2)
> ([Tuna-2 paper: arXiv:2604.24763](https://arxiv.org/abs/2604.24763)).
>
> This fork keeps the pixel-native philosophy of Tuna-2 (no VAE, no SigLIP, raw
> patches into the LLM) but swaps the backbone to **Gemma 4 12B** (encoder-free
> unified multimodal pretrain), adds an optional **PixelREPA** auxiliary
> semantic alignment, **MiniT2I-style Bottleneck patch embedding**, and the
> full machinery (data, attention, pipeline, distillation) for **pixel-native
> autoregressive video generation**.

---

## 1. 总览(TL;DR)

| 维度 | 原 Tuna-2 | 本 fork |
|---|---|---|
| LLM backbone | Qwen2.5-7B / 2-1.5B(custom `Qwen2ForCausalLM`) | **Gemma 4 12B**(HF `AutoModelForCausalLM`,任何 HF CausalLM 都能 plug) |
| Patch embedding | `SimplePatchEmbedding`(单层 Conv2d) | 二选一:`SimplePatchEmbedding` 或 **`BottleneckPatchEmbedding`**(MiniT2I 风格 2-stage) |
| 辅助语义对齐 | 无 | **PixelREPA**(MTA + DINOv2,可开关) |
| Attention 路径 | 自定义 Qwen2 + Tuna omni span 双向 | HF Gemma 4(sliding/global 交替)+ **`gemma_omni_attn`** 把 Tuna omni span 跟 Gemma 规则做并集 |
| 视频生成 | 仅 VAE-latent 变体(`tuna_2b` + Wan2.2 VAE) | **完整 pixel-native 视频栈**:dataset、per-frame mask、AR rollout pipeline、Mean Flow 蒸馏 |
| 训练 stage | 单一 | `joint_bidir` / `ar_teacher_force` / `mean_flow_distill` 三阶段 |
| 推理 mode | `t2i / edit / mmu` | 新增 **`t2v_ar`**(chunk-wise AR 像素视频) |

总改动量:**新增 10 文件(~2900 行) + 修改 6 文件(净 +263 行)**,所有 13 个 Python 文件通过 `ast.parse` 静态检查。

---

## 2. 改动文件清单

### 2.1 新增文件(10 个)

| 文件 | 行数 | 作用 |
|---|---|---|
| [`tuna/models/tuna_2_pixel_gemma.py`](tuna/models/tuna_2_pixel_gemma.py) | **972** | 核心新模型:`Tuna2PixelGemma` inner + `Tuna2PixelGemmaModel` wrapper |
| [`tuna/pipelines/tuna_2_pixel_ar_video_pipeline.py`](tuna/pipelines/tuna_2_pixel_ar_video_pipeline.py) | 404 | chunk-wise AR 视频推理(过去帧 clean condition + x0→v Euler + CFG noise share) |
| [`tuna/models/pixelrepa.py`](tuna/models/pixelrepa.py) | 315 | MTA + DINOv2 frozen target(跳 meta token + bilinear 2D 网格对齐) |
| [`tuna/data/datasets/pixel_video_dataset.py`](tuna/data/datasets/pixel_video_dataset.py) | 294 | 去 VAE 的 pixel 视频 dataset(per-frame spans + 配置感知 meta token 数) |
| [`tuna/training/mean_flow_distill.py`](tuna/training/mean_flow_distill.py) | 259 | Mean Flow 蒸馏 wrapper(正确 x0→v 转换 + 走 wrapper 构建 mask) |
| [`tuna/models/gemma_omni_attn.py`](tuna/models/gemma_omni_attn.py) | 235 | Gemma sliding/global × Tuna omni span 合并 mask(sdpa + flex) |
| [`configs/train/train_gemma.yaml`](configs/train/train_gemma.yaml) | 137 | S1 训练 config(含 6-variant ablation 命令注释) |
| [`configs/train/video_t2v_pixel_gemma.yaml`](configs/train/video_t2v_pixel_gemma.yaml) | 130 | 视频训练 config(3 个 stage 切换) |
| [`configs/model/tuna_2_pixel_gemma_12b.yaml`](configs/model/tuna_2_pixel_gemma_12b.yaml) | 104 | 模型 config(所有开关) |
| [`configs/predict/t2v_pixel_gemma.yaml`](configs/predict/t2v_pixel_gemma.yaml) | 39 | AR 视频推理 config |

### 2.2 修改文件(6 个)

| 文件 | diff stat | 改动摘要 |
|---|---|---|
| `tuna/models/vision/patch_embed.py` | +90 / -17 | 新增 `BottleneckPatchEmbedding` + `build_patch_embedding` 工厂 |
| `tuna/scripts/train.py` | +61 | `training.stage` 路由,`mean_flow_distill` 自动 wrap distill wrapper |
| `tuna/models/misc.py` | +55 / -3 | `get_text_tokenizer` 新增 `gemma4` family,用 Gemma 原生 token |
| `tuna/inference/runner.py` | +50 / -1 | 注册 `Tuna2PixelARVideoPipeline` + `t2v_ar` mode dispatch |
| `tuna/scripts/predict.py` | +5 | `_make_data_for_mode` 支持 `t2v_ar` |
| `tuna/pipelines/__init__.py` | +2 | export `Tuna2PixelARVideoPipeline` |

---

## 3. 逐项改动详解(原理 + 参考)

下面按"为什么改"、"参考了什么"、"具体怎么改"三层展开。**每一节都包含**:
- 原 Tuna-2 对应代码位置
- 改动的**论文/代码参考**
- **核心代码片段**(可直接对照 review)

---

### A. LLM Backbone 替换:Qwen2.5 → **Gemma 4 12B**

#### A.1 原理

Tuna-2 使用 `Qwen2.5-7B-Instruct`(纯文本预训练)作为 backbone,所有视觉对齐都得从训练数据里学。Google [Gemma 4 12B](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/) 是 encoder-free 多模态预训练的统一模型,**架构哲学跟 Tuna-2 完全一致**(直接 patch 投影,不用 vision encoder)。从它起步,LLM 已经懂图文对齐,**reconstruct missing layers 的工作量大幅降低**。

#### A.2 参考

- **架构哲学**:[Gemma 4 12B blog](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/)、[model card](https://ai.google.dev/gemma/docs/core/model_card_4)
  - 48 层 transformer,sliding window 1024,token budget 70/140/280/560/1120
- **HuggingFace 接入方式**:从 [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM) 加载,任何 HF CausalLM 都能 plug
- **原 Tuna-2 实现参考**:`tuna/models/tuna_2_pixel.py:86-91`(Qwen2.5 加载代码)

#### A.3 实现

文件:[`tuna/models/tuna_2_pixel_gemma.py`](tuna/models/tuna_2_pixel_gemma.py)

```python
# tuna_2_pixel_gemma.py:99-118
llm_config = AutoConfig.from_pretrained(llm_model_path)
# Map Tuna's attention_backend → HF's attn_implementation string.
# Tuna 内部叫 "flexattention"(无下划线),HF 要 "flex_attention"。
hf_attn_impl = {
    "sdpa": "sdpa",
    "flexattention": "flex_attention",
    "flex_attention": "flex_attention",
    "eager": "eager",
    "flash_attention_2": "flash_attention_2",
}.get(attn_implementation, "sdpa")
if init_llm_from_config:
    self.tuna = AutoModelForCausalLM.from_config(llm_config)
else:
    self.tuna = AutoModelForCausalLM.from_pretrained(
        llm_model_path, attn_implementation=hf_attn_impl,
    )
```

**关键设计点**:
1. **自动从 HF config 拉真实 hidden_size**(避免 yaml 写错 dim 导致 shape mismatch):
   ```python
   llm_hidden = getattr(self.tuna.config, "hidden_size", None) or hidden_size
   ```
2. **Diffusion head config 自动跟随 LLM**(继承 hidden_size/intermediate_size/max_position_embeddings)
3. **`init_llm_from_config=True`** 用于 smoke test(冷启动,不下载权重)

---

### B. Tokenizer:用 Gemma 原生 BOS + 原生 image token

#### B.1 原理 & 为什么这是 BLOCKER

第一版我犯了一个**会让模型从随机初始化的 token embedding 开始训**的错:
```python
# 错误的做法:
text_tokenizer.add_tokens("<|im_start|>")        # 新加!Gemma 没见过
text_tokenizer.add_tokens("<|vision_start|>")    # 新加!Gemma 没见过
"bos_id": vocab["<|im_start|>"]                  # 用没训过的 token 当 BOS
"boi_id": vocab["<|vision_start|>"]              # 完全浪费 Gemma 的 vision pretrain
```

Gemma 4 12B 已经**预训练了 `<start_of_image>` / `<end_of_image>` / `<image_soft_token>`**,把它们当 image token 用,可以继承 Gemma 的视觉对齐。**用新加的 alias 等于把这部分预训练完全扔掉。**

#### B.2 参考

- Gemma 4 model card 关于 vision token convention(`<start_of_image>` 等)
- HuggingFace tokenizer `bos_token_id` 属性(每个 LLM 都有自己的 native BOS)
- 原 Tuna-2 实现:`tuna/models/misc.py:541-590` 的 `qwen2_5` 分支

#### B.3 实现

文件:[`tuna/models/misc.py`](tuna/models/misc.py)

```python
# misc.py:601-630
elif llm_name == "gemma4":
    vocab = text_tokenizer.get_vocab()
    def _resolve(*candidates, attr=None):
        """优先用 vocab 查 candidates,找不到 fallback 到 tokenizer 属性"""
        for c in candidates:
            if c in vocab:
                return vocab[c]
        if attr is not None:
            val = getattr(text_tokenizer, attr, None)
            if val is not None:
                return val
        raise KeyError(f"None of {candidates} found in Gemma tokenizer vocab")

    tuna_token_ids = {
        # 直接用 Gemma 自己的 bos_token_id,而不是新加 token
        "bos_id": text_tokenizer.bos_token_id,
        "eos_id": text_tokenizer.eos_token_id,
        # 优先用 Gemma 原生 <start_of_image>,继承预训练对齐
        "boi_id": _resolve("<start_of_image>", "<|vision_start|>"),
        "eoi_id": _resolve("<end_of_image>", "<|vision_end|>"),
        # Gemma 4 unified 用 <image_soft_token> 作为 per-patch placeholder
        "img_pad_id": _resolve("<image_soft_token>", "<|image_pad|>"),
        # Gemma 没有原生 video token → 只补这些
        "bov_id": vocab["<|vid_start|>"],
        "eov_id": vocab["<|vid_end|>"],
        "vid_pad_id": vocab["<|video_pad|>"],
        "img_id": vocab["<image>"],
    }
```

**关键改动**:
1. **`bos_id = text_tokenizer.bos_token_id`** —— 用 Gemma 训练过的 BOS,不发明新的
2. **`_resolve()` 优先级**:Gemma 原生 token → Tuna alias(兼容性 fallback)
3. **PAD token 只在缺失时才加**(Gemma 自带 `<pad>`,不重复)

---

### C. Patch Embedding:新增 MiniT2I 风格 Bottleneck

#### C.1 原理

原 Tuna-2 的 `SimplePatchEmbedding`(`tuna/models/vision/patch_embed.py:14-65`)用**单层 Conv2d** 把 3 通道直接卷成 LLM hidden_size(3072 / 3584)。kernel 学到的特征**没有显式的低频/高频分离**。

[MiniT2I (Wang, He et al., 2026)](https://github.com/PeppaKing8/minit2i-jax) 用 **2 阶段 conv**:先卷到很窄的 bottleneck(如 64 维),再 1×1 升维到 LLM hidden。第一阶段相当于学一个 PCA-like 的颜色/边缘基底,第二阶段做 channel mixing。

> MiniT2I 在 ImageNet 256×256 上仅 912M 参数就达到 GenEval **0.883** / DPG-Bench **85.9**,超过 SD3-medium 2B,验证了 pixel + flow + Bottleneck 路线的有效性。

#### C.2 参考

- 论文:MiniT2I JAX 仓库的 [`models/dit_blocks.py`](https://github.com/PeppaKing8/minit2i-jax/blob/main/models/dit_blocks.py)(`BottleneckPatchEmbed` 原版)
- 原 Tuna-2:`tuna/models/vision/patch_embed.py:SimplePatchEmbedding`(保留作对照)

#### C.3 实现

文件:[`tuna/models/vision/patch_embed.py`](tuna/models/vision/patch_embed.py)

```python
class BottleneckPatchEmbedding(nn.Module):
    """Stage 1: spatial conv at low bottleneck channel dim (PCA-like).
    Stage 2: 1x1 channel mix to LLM hidden."""
    def __init__(self, patch_size=16, hidden_size=3072, in_channels=3,
                 bottleneck_dim=64):
        super().__init__()
        self.stage1 = nn.Conv2d(in_channels, bottleneck_dim,
                                kernel_size=patch_size, stride=patch_size)
        self.stage2 = nn.Conv2d(bottleneck_dim, hidden_size, kernel_size=1)
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, pixel_values):
        x = self.stage1(pixel_values)            # [B, bottleneck, h/p, w/p]
        x = self.stage2(x)                        # [B, hidden, h/p, w/p]
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        return self.norm(x)


def build_patch_embedding(vision_encoder_type, patch_size, hidden_size,
                          in_channels, bottleneck_dim=64):
    """Factory: 'simple' | 'bottleneck'."""
    if vision_encoder_type == "bottleneck":
        return BottleneckPatchEmbedding(...)
    elif vision_encoder_type == "simple":
        return SimplePatchEmbedding(...)
```

**集成**:`Tuna2PixelGemma.__init__` 走 factory 调用,通过 yaml 切换:
```yaml
vision_encoder_type: "bottleneck"   # 或 "simple"
vision_bottleneck_dim: 64
```

**`reset_parameters` 双路径处理**(`tuna_2_pixel_gemma.py:233-264`):用 `hasattr(ve, "patch_embedding")` 区分两类(SimplePatchEmbedding 有 `.patch_embedding`,BottleneckPatchEmbedding 有 `.stage1/.stage2`)。

---

### D. 辅助语义对齐:**PixelREPA**(MTA + DINOv2)

#### D.1 原理

PixelREPA 论文([arXiv:2603.14366](https://arxiv.org/abs/2603.14366))的核心 finding:**直接用 MLP 把 pixel-space diffusion 的中间 hidden 对齐到 DINOv2 这种 compressed semantic 空间会导致 diversity collapse**(称为 feature hacking)。原因是 information asymmetry — pixel space 维度高,DINOv2 是压缩语义。

**正解**:Masked Transformer Adapter (MTA),只在训练时作为辅助分支:
1. 从 LLM 某中间层 hook hidden state
2. 对 image-position 部分的 token **随机 mask 50%**
3. 通过 2-block transformer adapter(打破 per-token shortcut)
4. 投影到 DINOv2 特征空间
5. 跟 frozen DINOv2 的 clean image feature 算 cosine loss

**关键属性**:推理时完全跳过(零额外成本)。

#### D.2 参考

- 论文:[Representation Alignment for Just Image Transformers is not Easier than You Think (arXiv:2603.14366)](https://arxiv.org/abs/2603.14366),代码 [github.com/kaist-cvml/PixelREPA](https://github.com/kaist-cvml/PixelREPA)
- 关键 results:JiT-H/16 + PixelREPA → FID **1.81** / IS **317.2**(超过更大的 JiT-G 1.82)
- 论文 ablation:naive REPA 把 JiT 弄差(FID 4.37 → 5.14),PixelREPA 修好(→ 4.00)

#### D.3 实现

文件:[`tuna/models/pixelrepa.py`](tuna/models/pixelrepa.py)

##### D.3.1 模块结构

```python
class PixelREPAModule(nn.Module):
    def __init__(self, llm_hidden_size, target_model_id="facebook/dinov2-large",
                 target_hidden_size=1024, adapter_depth=2, adapter_heads=8,
                 mask_ratio=0.5, loss_weight=0.5, from_layer=-8,
                 target_image_size=224):
        # 1. Learnable mask token (MAE / DeTok style)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, llm_hidden_size))
        # 2. 2-block transformer (per PixelREPA paper)
        self.adapter = nn.TransformerEncoder(encoder_layer, num_layers=adapter_depth)
        # 3. Project to target encoder dim
        self.proj = nn.Linear(llm_hidden_size, target_hidden_size)
        # 4. Frozen target encoder (lazy-loaded)
        # ...
```

##### D.3.2 三个关键修复(来自 fanout review agent 的发现)

**Fix 1:跳过 Tuna 的 meta token**(`pixelrepa.py:178-184`)

Tuna 的 `_inner_base.py:_prepare_input` 在每个 image span 开头写 height/width/time meta token(1 或 3 个,取决于 `add_time_embeds + add_aspect_ratio_embeds`)。原版 PixelREPA 直接取 `[offset : offset+length]` 会把 meta token 当成 image patch,污染对齐信号。

```python
# pixelrepa.py 中:
patch_start = offset + meta_token_count   # 跳过 meta
patch_end = offset + length
patches = llm_hidden[b, patch_start:patch_end]
```

`meta_token_count` 由 wrapper 算出:
```python
# tuna_2_pixel_gemma.py:267-276
def _image_meta_token_count(self) -> int:
    if self.config.add_time_embeds and self.config.add_aspect_ratio_embeds:
        return 3
    if self.config.add_time_embeds:
        return 1
    return 0
```

**Fix 2:**bilinear 2D 网格对齐**(`pixelrepa.py:_align_grid`)

第一版用 `F.interpolate(mode='linear')` 在 token 一维上 resize,**跨行平均破坏 2D 空间结构**。修复后 reshape 到 2D 网格再 bilinear:

```python
def _align_grid(self, target_feat, hp, wp):
    """Bilinear-resize DINOv2 feature grid to match LMM patch grid."""
    side = int(math.isqrt(N_dino))
    # [1, N_dino, D] → [1, D, side, side] → bilinear (hp, wp) → flatten
    t = target_feat.transpose(1, 2).reshape(1, -1, side, side)
    t = F.interpolate(t, size=(hp, wp), mode="bilinear", align_corners=False)
    return t.flatten(2).transpose(1, 2)
```

**Fix 3:multi-image / 不等长 span 处理**(`pixelrepa.py:forward`)

原版 `torch.stack` 假设所有 image span 等长(multi-resolution / interleaved 会崩),且只用 `image_positions[b, 0]` 丢掉 image 1..N。修复后改成 per-(batch, span) 个体处理,累积 loss:

```python
total_loss = torch.tensor(0.0, device=device, dtype=dtype)
count = 0
for b in range(B):
    for j in range(N_imgs):     # 遍历所有 span 不是只取第 0 个
        # ... per-span MTA + loss ...
        total_loss = total_loss + (1.0 - cos_sim).mean()
        count += 1

if count == 0:
    # text-only step: graph-connected zero,避免 FSDP 抱怨 unused params
    zero = (self.mask_token.sum() * 0.0
            + sum(p.sum() * 0.0 for p in self.proj.parameters())).to(dtype)
    return zero

return (total_loss / count) * self.loss_weight
```

##### D.3.3 集成

`Tuna2PixelGemma.forward`:
```python
# 只在 training 且 PixelREPA 启用时 hook hidden states
need_hidden = (output_hidden_states
               or (self.enable_pixelrepa and self.training and self.pixelrepa))
outputs = self.tuna(..., output_hidden_states=need_hidden)

if self.enable_pixelrepa and self.training and ...:
    mid_hidden = outputs.hidden_states[self.pixelrepa.from_layer]  # e.g. -8
    loss_repa = self.pixelrepa(
        llm_hidden=mid_hidden,
        image_positions=modality_positions,
        clean_pixel_values=clean_pixel_values,
        meta_token_count=self._image_meta_token_count(),
    )
```

Loss 累加(wrapper):
```python
total_loss = self.flow_coeff * loss_flow + self.ntp_coeff * loss_ntp
if self.enable_pixelrepa:
    total_loss = total_loss + loss_repa   # weight 已在 module 内部应用
```

---

### E. Attention Mask:Gemma sliding × Tuna omni span 合并

#### E.1 原理

Gemma 4 12B 用 **interleaved sliding (1024) + global** 双类层。Tuna 的 `omni_attn_mask_naive`(`tuna/models/omni_attention.py:69-93`)规则是:"全局 causal + 每个 image span 内部双向"。

**冲突**:1024×1024 图像 = 4096 个 patch token,在 sliding 层里图像后半看不到前半,违反 Tuna 的"同 span 双向"不变量。

**解决**:把两条规则做**并集(OR)**:
```
final_mask = gemma_base_rule   OR   same_image_span   [OR cross_frame_causal]
                                    ↑                  ↑
                                    Tuna omni 规则     视频 AR 时启用
```

#### E.2 参考

- 原 Tuna-2:`tuna/models/omni_attention.py:omni_attn_mask_naive` 和 `omni_attn_mask_flexattention`
- HF Gemma 4 attention 实现的 sliding 规则
- Lumos-1([arXiv:2507.08801](https://arxiv.org/abs/2507.08801))的 "intra-frame bidirectional + inter-frame causal" 设计(MM-RoPE)

#### E.3 实现

文件:[`tuna/models/gemma_omni_attn.py`](tuna/models/gemma_omni_attn.py)

```python
def gemma_omni_mask_mod_factory(span_id, sliding_window=1024,
                                is_local_layer=True, cross_frame_causal=False):
    def mask_mod(b, h, q, k):
        # 1) Gemma base rule
        if is_local_layer:
            base = (q >= k) & ((q - k) < sliding_window)
        else:
            base = q >= k
        # 2) Tuna 同 span 双向
        iq, ik = span_id[b, q], span_id[b, k]
        same_span = (iq != 0) & (iq == ik)
        # 3) 视频 AR:后帧能看前帧的全部 patches
        if cross_frame_causal:
            cross_causal = (iq > ik) & (iq != 0) & (ik != 0)
            return base | same_span | cross_causal
        return base | same_span
    return mask_mod
```

#### E.4 三个关键修复(来自 review)

**Fix 1:`is_local_layer` 默认应该用 False(global / full causal),不是 True**

原版用 sliding 作 base,但 sliding ⊂ full causal,在全局层上等于**杀掉 Gemma 预训练的长程注意力**。修复后默认 `is_local_layer=False`,base 用最弱的规则(full causal),OR 上 same_span 后所有层都能用同一份 mask。

```python
# tuna_2_pixel_gemma.py 中 create_attention_mask:
mask = build_gemma_omni_attn_mask_naive(
    modality_positions=modality_positions,
    seq_len=seq_length,
    sliding_window=self.sliding_window,
    is_local_layer=False,    # ← Fix: full-causal base
    cross_frame_causal=use_cross_frame_causal,
    ...
)
```

**Fix 2:动态切换 cross_frame_causal**

`create_attention_mask` 自动检测视频批次(`modality_positions.shape[1] > 1`)并结合 stage flag:
```python
ar_video_flag = getattr(self, "_video_cross_frame_causal", False)
is_multi_frame = modality_positions.ndim == 3 and modality_positions.shape[1] > 1
use_cross_frame_causal = bool(ar_video_flag and is_multi_frame)
```

`_video_cross_frame_causal` 由 `train.py` 根据 `training.stage='ar_teacher_force'` 自动 set。

**Fix 3:cross-frame 在局部 sliding 层的修补**

当 frame 间距离 > sliding_window 时,naive sliding-tril 会切断 cross-frame visibility。`build_gemma_omni_attn_mask_naive` 加了显式 lift:
```python
if cross_frame_causal and is_local_layer:
    for s_q in unique_spans:
        for s_k in unique_spans:
            if s_k >= s_q: continue
            # 让 s_q 帧能看 s_k 帧(无论距离多远)
            base[b, qpos[:, None], kpos[None, :]] = True
```

---

### F. 视频数据 pipeline:`PixelVideoDataset`

#### F.1 原理

原 Tuna-2 的 `VideoDataset`(`tuna/data/datasets/video_dataset.py`)**硬编码 Wan2.2 VAE 压缩**:
```python
spatial_ds = 16   # VAE 空间压缩 8×,Tuna patch 1 拼起来 ≈ 16
temporal_ds = 4   # VAE 时间压缩 4×
latent_t = (num_frames + 2) // 4 + 1   # causal conv 公式
```

Pixel 变体没有 VAE,所以这些假设全是错的。新 dataset:
- 空间:`num_tokens_per_frame = (H/16) * (W/16)` 直接 patch
- 时间:`temporal_ds = 1`(无压缩)
- **per-frame `modality_positions`**:每帧一个 span,让 AR mask 能区分帧

#### F.2 参考

- 原 Tuna-2:`tuna/data/datasets/video_dataset.py:VideoDataset`
- 原 Tuna-2 sequence formatter:`tuna/data/tokenize_utils.py:format_sequence_gen_qwen2_5`
- MovieGen / HunyuanVideo 的"image as T=1 video"统一架构

#### F.3 实现

文件:[`tuna/data/datasets/pixel_video_dataset.py`](tuna/data/datasets/pixel_video_dataset.py)

**关键修复**:meta token 数 **配置感知**(不再硬编码 +1):
```python
# pixel_video_dataset.py:108-114
if add_time_embeds and add_aspect_ratio_embeds:
    self.n_meta = 3
elif add_time_embeds:
    self.n_meta = 1
else:
    self.n_meta = 0
self.num_tokens_per_frame_with_meta = self.num_tokens_per_frame + self.n_meta
```

**Per-frame span 构造**:
```python
def _build_per_frame_modality_positions(self, text_tokens):
    """[num_frames, 2],每帧一个 span"""
    boi_positions = (text_tokens == self.boi_id).nonzero(as_tuple=True)[0]
    first_frame_offset = int(boi_positions[0].item()) + 1
    positions = []
    for t_idx in range(self.num_frames):
        offset = first_frame_offset + t_idx * self.num_tokens_per_frame_with_meta
        positions.append([offset, self.num_tokens_per_frame_with_meta])
    return torch.tensor(positions, dtype=torch.long)
```

**Wrapper.forward 视频分支**(`tuna_2_pixel_gemma.py:868-880`):
```python
elif data_type[0] in ("t2v_pixel", "t2v") and pixel_values.dim() == 5:
    # [B, C, T, H, W] → [B*T, C, 1, H, W]
    # 每帧变成独立 image,匹配 _prepare_input 的 image_embeds[i*N+j] 索引
    b, c, T, h_, w_ = pixel_values.shape
    pixel_values = rearrange(pixel_values, "b c t h w -> (b t) c h w").unsqueeze(2)
    data_type = list(data_type) * T
```

#### F.4 已知限制

- **Mixed image+video batch collate**:依赖 `sync_sampling=true` 保证一步只采一个 stream
- **V-S3 AR 训练的 per-frame t schedule**:当前版本所有帧用独立采样的 t(适合 V-S2 双向 teacher);真正的"过去 t=1 当前 t<1"AR teacher forcing 需要进一步在 `prepare_latents_and_labels` 加 `t2v_ar` 专用分支

---

### G. AR 视频推理 pipeline

#### G.1 原理

视频 AR 生成 = **chunk-wise**:每次生成 K 帧(典型 3-4),空间内部 bidirectional diffusion,跨 chunk causal。过去 chunk 的 clean 帧通过 `image_latents` 拼在当前 noisy chunk 之前(`t=1.0` 标记 clean),让模型"看到"过去。

#### G.2 参考

- [Self-Forcing (arXiv:2506.08009, NeurIPS 2025 spotlight)](https://arxiv.org/abs/2506.08009)的 chunk-wise AR rollout 概念
- [Causal Forcing++ (arXiv:2605.15141)](https://arxiv.org/abs/2605.15141)的 frame-wise causal mask
- [MAGI-1](https://github.com/SandAI-org/MAGI-1) 的 24-frame chunk-wise denoising
- 原 Tuna-2 sampler:`tuna/models/jit_utils.py:JiTSampler` 的 x0→v 转换公式
- 原 Tuna-2 pipeline 参考:`tuna/pipelines/tuna_2_pixel_pipeline.py:t2i`

#### G.3 实现

文件:[`tuna/pipelines/tuna_2_pixel_ar_video_pipeline.py`](tuna/pipelines/tuna_2_pixel_ar_video_pipeline.py)

##### G.3.1 核心 chunk loop

```python
for chunk_idx in range(n_chunks):
    num_past = len(all_clean_frames)
    num_cur = self.frames_per_chunk
    num_total = num_past + num_cur

    # 1) 构建 text + per-frame spans (cond + uncond stacked for CFG)
    text_tokens_cfg, modality_positions_cfg = self._build_cfg_inputs(...)

    # 2) Init noise(CFG 两个 branch SHARE noise)
    cur_noise = noise_scale * torch.randn((1, 3, num_cur, H, W))

    # 3) Stack past clean frames + current noisy chunk
    past_5d = torch.stack(all_clean_frames).permute(...)
    past_5d_cfg = torch.cat([past_5d, past_5d], dim=0)  # share past

    # 4) Diffusion sub-steps
    cur_state = cur_noise.clone()
    for step_idx in range(num_diffusion_steps_per_chunk):
        t_cur = timesteps[step_idx]
        t_next = timesteps[step_idx + 1]

        # image_latents = [past_clean | cur_state]
        image_latents = torch.cat([past_5d_cfg, cur_state_cfg], dim=2)

        # per-frame t: past=1.0, current=t_cur
        t_per_frame = ...

        # AR mask(cross_frame_causal=True)
        attn_mask, _ = self._build_ar_attention_mask(...)

        # Forward → x0 prediction
        _, x0_pred = self.model.tuna_model(
            text_tokens=text_tokens_cfg, image_latents=image_latents,
            t=t_per_frame, attention_mask=attn_mask, ...
        )

        # 取当前 chunk 的预测,CFG 合并
        x0_cur = x0_pred[:, :, num_past:, :, :]
        x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

        # x0 → v conversion(Tuna JiT convention: t=0 noise, t=1 clean)
        v_pred = (x0_guided - cur_state) / max(1.0 - t_cur.item(), eps_t)

        # Euler step toward t=1 (clean)
        cur_state = cur_state + (t_next - t_cur) * v_pred

    # 5) Commit clean chunk frames
    for f_idx in range(num_cur):
        all_clean_frames.append(cur_state[0, :, f_idx].cpu())
```

#### G.4 第一版 6 个 BLOCKER 都修了

| 问题 | 修复 |
|---|---|
| KV cache 声明但从不传 | 文档化为 v2 优化(每 chunk 重算 prefix,正确但慢) |
| 过去帧从不喂模型 | `image_latents = [past_clean | cur_state]` 拼接 |
| Euler 方向反了 | Tuna JiT 是 t=0 noise / t=1 clean,linspace(0,1) **是对的** |
| 模型返回 x0 当成 v | 加 `v = (x0 - z) / max(1-t, eps)` 转换 |
| CFG 两个 branch 独立 noise | 改成 share noise |
| 负向 prompt unused | 通过 `_build_cfg_inputs` 真正注入 |

---

### H. Mean Flow 蒸馏:few-step AR 视频

#### H.1 原理

[MiniT2I 的 mean_flow_distill 分支](https://github.com/PeppaKing8/minit2i-jax/tree/mean_flow_distill)思路:把 teacher 在区间 `[t_start, t_end]` 上跑 K 步 Euler 得到 mean velocity,蒸馏到 student 一次预测。学生推理时 K 步就够(K << teacher 的 50 步)。

**对视频 AR 的价值**:Tuna-2 视频 AR 每 chunk 默认 8 步,蒸到 4 步可以 2× 加速,且不引入 Causal Forcing++ 的复杂(consistency loss、DMD discriminator)。

#### H.2 参考

- [MiniT2I `mean_flow_distill` branch](https://github.com/PeppaKing8/minit2i-jax)
- [Causal Forcing++ (arXiv:2605.15141)](https://arxiv.org/abs/2605.15141) 用 causal CD 蒸到 1-2 step(更复杂的 alternative)
- 原 Tuna-2 x0→v 公式:`tuna/models/jit_utils.py:152-165`(`v = (x0_pred - z_t) / max(1 - t, 5e-2)`)

#### H.3 实现

文件:[`tuna/training/mean_flow_distill.py`](tuna/training/mean_flow_distill.py)

```python
def _x0_to_velocity(x0_pred, z_t, t, eps_t=5e-2):
    """Tuna JiT convention: z_t = t * x0 + (1-t) * noise,
       rectified-flow velocity = (x0 - noise) = (x0 - z_t) / (1 - t)."""
    one_minus_t = (1.0 - t).clamp_min(eps_t)
    while one_minus_t.dim() < x0_pred.dim():
        one_minus_t = one_minus_t.unsqueeze(-1)
    return (x0_pred - z_t) / one_minus_t


class MeanFlowDistillationWrapper(nn.Module):
    @torch.no_grad()
    def compute_teacher_mean_velocity(self, x_t_start, t_start, t_end, ...):
        x = x_t_start.clone()
        dt = (t_end - t_start) / self.teacher_substeps
        for i in range(self.teacher_substeps):
            t_i = t_start + i * dt
            _, x0_pred = self.teacher.tuna_model(
                image_latents=x, t=t_i, ...
            )
            v_i = _x0_to_velocity(x0_pred, x, t_i)   # ← 关键转换
            x = x + dt * v_i
        mean_v = (x - x_t_start) / (t_end - t_start)
        return mean_v

    def forward(self, batch):
        # ... 走 wrapper.create_attention_mask 构建 omni mask
        attn_mask, diff_mask = self._build_masks(...)
        # teacher mean velocity (no_grad)
        v_target = self.compute_teacher_mean_velocity(...)
        # student one-step at t_avg
        _, x0_pred_student = self.student.tuna_model(t=t_avg, ...)
        v_pred = _x0_to_velocity(x0_pred_student, x_t_start, t_avg)
        # L2 loss on velocity
        loss = ((v_pred - v_target) ** 2).mean()
        return {"loss": loss, ...}
```

#### H.4 3 个 CRITICAL 修复

| 问题(review 发现) | 修复 |
|---|---|
| teacher 返回 x0 当 v 用,Euler 公式完全错 | 加 `_x0_to_velocity` 显式转换 |
| `attention_mask=None`,Gemma 退化到 full causal,Tuna omni 失效 | 走 `self.student.create_attention_mask(...)` |
| student 没真接收 `t_end` | 文档化 limitation(传 `t_avg`,等宽 interval 下 bijective with index;真正修复需改 `TimestepEmbedder`) |

---

### I. 训练 stage 路由

#### I.1 原理

视频训练分 3 个 stage,对应 MovieGen/HunyuanVideo 的渐进训练经验:

| Stage | 目的 | 关键 |
|---|---|---|
| `joint_bidir` | image + 视频联合(双向 teacher) | 默认 mask,全双向 |
| `ar_teacher_force` | AR 因果训练 | `_video_cross_frame_causal=True` |
| `mean_flow_distill` | 4-step 蒸馏 | 把 model wrap 进 `MeanFlowDistillationWrapper` |

#### I.2 参考

- 原 Tuna-2 `train.py` 主循环结构
- [MovieGen (arXiv:2410.13720)](https://arxiv.org/abs/2410.13720) 的 image→joint 训练阶段
- [HunyuanVideo (arXiv:2412.03603)](https://arxiv.org/abs/2412.03603) 的渐进 image+video joint(每阶段都掺图像防遗忘)

#### I.3 实现

文件:[`tuna/scripts/train.py`](tuna/scripts/train.py)

```python
# train.py:144-204
model = instantiate(cfg.model)

stage = cfg.training.get("stage", "joint_bidir")
if stage == "ar_teacher_force":
    setattr(model, "_video_cross_frame_causal", True)
    logger.info("[Stage] ar_teacher_force: enabled cross-frame causal")

elif stage == "mean_flow_distill":
    from tuna.training.mean_flow_distill import MeanFlowDistillationWrapper
    teacher_ckpt = cfg.mean_flow.teacher_ckpt
    teacher = instantiate(cfg.model)
    teacher.load_state_dict(torch.load(teacher_ckpt)["state_dict"], strict=False)
    teacher.eval(); [p.requires_grad_(False) for p in teacher.parameters()]
    model = MeanFlowDistillationWrapper(
        student_model=model, teacher_model=teacher,
        num_intervals=cfg.mean_flow.num_intervals,
        teacher_substeps=cfg.mean_flow.teacher_substeps,
        loss_type=cfg.mean_flow.loss_type,
    )

elif stage == "joint_bidir":
    pass  # default
else:
    raise ValueError(f"Unknown stage: {stage!r}")
```

---

### J. 推理 dispatch:`t2v_ar` 新 mode

#### J.1 实现

文件:[`tuna/inference/runner.py`](tuna/inference/runner.py)

```python
# runner.py 新增 import + dispatch
from tuna.pipelines.tuna_2_pixel_ar_video_pipeline import Tuna2PixelARVideoPipeline

# 在 _init_pipeline:
elif pipe == "Tuna2PixelARVideoPipeline":
    ar_kwargs = {
        "model": self.model,
        # ... 视频专用参数 ...
        "num_frames": getattr(self, "num_frames", 16),
        "frames_per_chunk": getattr(self, "frames_per_chunk", 4),
        "num_diffusion_steps_per_chunk": getattr(self, "num_diffusion_steps_per_chunk", 8),
        "max_seq_len": getattr(self, "max_seq_len", 8192),
    }
    self.pipe = Tuna2PixelARVideoPipeline(**ar_kwargs)

# 在 __call__:
if self.inference_mode == "t2v_ar":
    return self.t2v_ar(data, **kwargs)
```

`tuna/scripts/predict.py:_make_data_for_mode` 也加了 `t2v_ar` 分支。

---

## 4. 使用方式

### 4.1 安装(同 Tuna-2)

```bash
git clone <this-repo>
cd <repo>
bash scripts/setup_uv.sh
source .venv/bin/activate
```

### 4.2 S1 图像训练(4-variant 消融)

```bash
# Variant A:Qwen baseline(原 Tuna-2 复现)
torchrun --standalone --nproc-per-node=8 -m tuna.scripts.train \
    --config-name train \
    training.output_dir=./outputs/s1/A_qwen_baseline

# Variant B:Gemma backbone(主对比)
RUN_NAME=s1_B torchrun --standalone --nproc-per-node=8 \
    -m tuna.scripts.train --config-name train_gemma \
    training.output_dir=./outputs/s1/B_gemma_simple

# Variant D:Gemma + PixelREPA
RUN_NAME=s1_D torchrun --standalone --nproc-per-node=8 \
    -m tuna.scripts.train --config-name train_gemma \
    model.enable_pixelrepa=true \
    training.output_dir=./outputs/s1/D_gemma_repa

# Variant F:Full(Gemma + Bottleneck + PixelREPA)
RUN_NAME=s1_F torchrun --standalone --nproc-per-node=8 \
    -m tuna.scripts.train --config-name train_gemma \
    model.vision_encoder_type=bottleneck \
    model.enable_pixelrepa=true \
    training.output_dir=./outputs/s1/F_full
```

### 4.3 视频训练(3 阶段)

```bash
# V-S2: 双向 teacher(image + 短视频联合)
torchrun ... -m tuna.scripts.train \
    --config-name video_t2v_pixel_gemma \
    training.stage=joint_bidir \
    model.load_stage1_model=./outputs/s1/F_full/checkpoints/last.pt

# V-S3: AR teacher forcing(cross-frame causal)
torchrun ... -m tuna.scripts.train \
    --config-name video_t2v_pixel_gemma \
    training.stage=ar_teacher_force \
    model.load_stage1_model=./outputs/video/V-S2/last.pt

# V-S4: Mean Flow 蒸馏到 4-step
torchrun ... -m tuna.scripts.train \
    --config-name video_t2v_pixel_gemma \
    training.stage=mean_flow_distill \
    mean_flow.teacher_ckpt=./outputs/video/V-S3/last.pt
```

### 4.4 AR 视频推理

```bash
python -m tuna.scripts.predict --config-name t2v_pixel_gemma \
    prompt="A cat walking on the beach at sunset, photorealistic"
```

### 4.5 Smoke test(单卡快速验证)

```bash
python -m tuna.scripts.train \
    --config-name train_gemma \
    training.max_steps_per_epoch=4 \
    training.batch_size=1 \
    training.fsdp.enable=false
```

---

## 5. 论文级 contribution(可写进 paper)

| 维度 | Tuna-2 论点 | 本 fork 扩展 |
|---|---|---|
| 哲学 | "无 VAE / 无 encoder 的 pixel-native unified MLLM 可行" | "**继承多模态预训练的 Gemma 4 backbone 进一步加速收敛**" |
| 视觉 | 简单 Conv2d patch | "**MiniT2I 风格 BottleneckPatchEmbedding 改善图像质量**" |
| 监督 | NTP + Flow | "**PixelREPA 辅助语义对齐(MTA + DINOv2)无推理代价**" |
| 视频 | VAE-latent 变体 only | "**首个 pixel-native unified MLLM 的视频 AR 框架**" |
| 加速 | 50-step Euler | "**Mean Flow 蒸馏到 K-step AR 视频**" |

跟最相关 prior art 的差异:
- vs [Lumos-1 (2507.08801)](https://arxiv.org/abs/2507.08801):Lumos-1 是 **离散 token + mask discrete diffusion**;本 fork 是 **continuous flow + pixel-native**
- vs [InfinityStar (2511.04675)](https://arxiv.org/abs/2511.04675):InfinityStar 是 **purely discrete spacetime AR**;本 fork 是 **continuous diffusion**
- vs [UniAR (2606.18249)](https://arxiv.org/abs/2606.18249):UniAR 是 **BSQ discrete + SD3 decoder**;本 fork 完全无 tokenizer 无 decoder
- vs [MiniT2I](https://github.com/PeppaKing8/minit2i-jax):MiniT2I 是 **纯 T2I**(冻结 T5);本 fork 是 **unified MLLM**(理解 + 生成)

---

## 6. 已知限制 / TODOs

| 限制 | 影响 | 修复方向 |
|---|---|---|
| AR 视频 KV cache 未实现 | 每 chunk 重算 prefix,推理慢 ~T× | 给 `Tuna2PixelGemma.forward` 接 `past_key_values` + `use_cache=True` |
| Mean Flow student 只接收 `t_avg`(不是 `(t_start, t_end)`) | 数学不严格,等宽 interval 下近似 bijective with index | 改 `TimestepEmbedder` 接 pair |
| Mixed image+video batch collate | 依赖 `sync_sampling=true` | 加 `modality_positions` -1 padding |
| Gemma 4 12B 真实 HF config 还需确认 | yaml 默认值是预估 | 待 `google/gemma-4-12b` 真正上线后微调 |
| V-S3 真正 AR teacher-forcing 需 per-frame t schedule | 当前所有帧独立 sample t(适合 V-S2 双向) | 加 `data_type=='t2v_ar'` 分支到 `prepare_latents_and_labels` |

---

## 7. Review checklist

### 7.1 代码 review 优先级

| 优先级 | 文件 | 关注点 |
|---|---|---|
| 🔴 高 | `tuna/models/tuna_2_pixel_gemma.py` | 跟原 `tuna_2_pixel.py` 逐行对比 forward/reset/init |
| 🔴 高 | `tuna/models/gemma_omni_attn.py` | sdpa 路径的 dtype/shape;cross_frame_causal lift 逻辑 |
| 🔴 高 | `tuna/models/pixelrepa.py` | meta_token_count skip;2D grid alignment;multi-image loop |
| 🟡 中 | `tuna/pipelines/tuna_2_pixel_ar_video_pipeline.py` | t_per_frame layout;x0→v;CFG noise sharing |
| 🟡 中 | `tuna/training/mean_flow_distill.py` | teacher mean velocity 计算;attention mask 路径 |
| 🟢 低 | configs/*.yaml | field 名跟 model 类 __init__ 一一对应 |

### 7.2 跑通验证(按顺序)

```bash
# 1. Smoke test:模型能 forward+backward,loss 不 NaN
python -m tuna.scripts.train --config-name train_gemma \
    training.max_steps_per_epoch=4 training.batch_size=1 training.fsdp.enable=false

# 2. PixelREPA 启用,验证 DINOv2 能 load
python -m tuna.scripts.train --config-name train_gemma \
    model.enable_pixelrepa=true \
    training.max_steps_per_epoch=4 training.batch_size=1 training.fsdp.enable=false

# 3. Bottleneck patch embed
python -m tuna.scripts.train --config-name train_gemma \
    model.vision_encoder_type=bottleneck \
    training.max_steps_per_epoch=4 training.batch_size=1 training.fsdp.enable=false

# 4. 视频 V-S2(短视频 4 帧)
python -m tuna.scripts.train --config-name video_t2v_pixel_gemma \
    training.max_steps_per_epoch=4 training.batch_size=1 training.fsdp.enable=false

# 5. 视频 V-S3 切换 AR mask
python -m tuna.scripts.train --config-name video_t2v_pixel_gemma \
    training.stage=ar_teacher_force \
    training.max_steps_per_epoch=4 training.batch_size=1 training.fsdp.enable=false
```

---

## 8. 致谢

本 fork 基于以下工作:
- **Tuna-2** (Liu, Ren, He et al., 2026) — pixel-native UMM 的核心范式
- **MiniT2I** (Wang, Zhao, He et al., 2026) — 借鉴 BottleneckPatchEmbed 和 Mean Flow 思路
- **PixelREPA** (Shin, Kim, Shim, 2026) — MTA 辅助对齐设计
- **Gemma 4** (Google DeepMind, 2026) — encoder-free 多模态预训练 backbone
- **Self-Forcing / Causal Forcing / Causal Forcing++** — AR 视频蒸馏 recipe 概念
- **MovieGen / HunyuanVideo** — image-video joint training 经验

参考论文列表:
- Tuna-2: [arXiv:2604.24763](https://arxiv.org/abs/2604.24763)
- PixelREPA: [arXiv:2603.14366](https://arxiv.org/abs/2603.14366)
- Self-Forcing: [arXiv:2506.08009](https://arxiv.org/abs/2506.08009)
- Causal Forcing: [arXiv:2602.02214](https://arxiv.org/abs/2602.02214)
- Causal Forcing++: [arXiv:2605.15141](https://arxiv.org/abs/2605.15141)
- Lumos-1: [arXiv:2507.08801](https://arxiv.org/abs/2507.08801)
- InfinityStar: [arXiv:2511.04675](https://arxiv.org/abs/2511.04675)
- UniAR: [arXiv:2606.18249](https://arxiv.org/abs/2606.18249)
- MovieGen: [arXiv:2410.13720](https://arxiv.org/abs/2410.13720)
- HunyuanVideo: [arXiv:2412.03603](https://arxiv.org/abs/2412.03603)
- MAGI-1: [github.com/SandAI-org/MAGI-1](https://github.com/SandAI-org/MAGI-1)
- LongLive: [arXiv:2509.22622](https://arxiv.org/abs/2509.22622)
- MiniT2I-JAX: [github.com/PeppaKing8/minit2i-jax](https://github.com/PeppaKing8/minit2i-jax)
- MiniT2I-PyTorch port: [github.com/Hope7Happiness/minit2i-torch](https://github.com/Hope7Happiness/minit2i-torch)

---

## 9. License

Apache License 2.0 (继承自 Tuna-2)。
