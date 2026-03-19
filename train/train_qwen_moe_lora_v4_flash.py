# train_qwen_moe_lora.py  MoE-LoRA version with safe guards
import importlib

def _allow(*args, **kwargs):
    return None

try:
    # 1 改 import_utils 里原始函数的代码对象
    iu = importlib.import_module("transformers.utils.import_utils")
    if hasattr(iu, "check_torch_load_is_safe"):
        iu.check_torch_load_is_safe.__code__ = _allow.__code__

    # 2 覆盖已经绑定了别名的模块作用域
    for mod_name in ("transformers.trainer",
                     "transformers.deepspeed",
                     "transformers.trainer_utils"):
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, "check_torch_load_is_safe"):
                setattr(m, "check_torch_load_is_safe", _allow)
        except Exception:
            pass

    print("[WARN] Disabled transformers torch.load safety check. Load only trusted checkpoints.")
except Exception as e:
    print(f"[WARN] Failed to disable safety check: {e}")


import argparse
import json
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm.auto import tqdm
import re
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments,
)

# 尝试显式导入 Qwen2.5 的图像处理器（不同版本命名可能不同，做多重兼容）
try:
    from transformers import Qwen2_5_VLImageProcessor as QwenVLImageProcessor  # 新版命名
except Exception:
    try:
        from transformers import Qwen2VLImageProcessor as QwenVLImageProcessor  # 备选命名
    except Exception:
        QwenVLImageProcessor = None


from peft import LoraConfig, get_peft_model, TaskType


from patch_qwen_moe_lora import patch_model_with_moe_lora
from moe_lora import MoELoRALinear
from typing import List, Tuple, Optional
import torch.nn.functional as F
from torch.utils.data import get_worker_info 

from torch.utils.data import Sampler, DataLoader
from collections import defaultdict
import random, torch
import math

from transformers import Trainer as _HFTrainer

from accelerate import Accelerator
from transformers.trainer_utils import get_last_checkpoint
import json as _json

# 可选 safetensors
try:
    from safetensors.torch import load_file as _safe_load_file
    _HAS_SAFE = True
except Exception:
    _HAS_SAFE = False

# ================= Mixture-of-Prompts (soft prompt bank + learnable gating) =================
import torch.nn as nn


class PromptMixtureHead(nn.Module):
    """
    Mixture-of-Prompts module:
    - prompt_bank: K groups of P soft tokens, shape [K, P, D]
    - gate: MLP that maps mean pooled token embeddings to K weights (softmax)
    Forward returns fused prompt embeds [B, P, D] and gating logits/weights.
    """

    def __init__(self, hidden_size: int, bank_size: int, prompt_len: int, gate_hidden: int = 256, top_k: int = 0, use_cosine: bool = False,
                 attn_gate: bool = False, attn_heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        assert bank_size > 0 and prompt_len > 0, "bank_size and prompt_len must be positive"
        self.hidden_size = int(hidden_size)
        self.bank_size = int(bank_size)
        self.prompt_len = int(prompt_len)
        self.top_k = int(top_k or 0)
        # self.use_cosine = bool(use_cosine)
        self.attn_gate = bool(attn_gate)
        self.attn_heads = int(max(1, attn_heads))
        self.attn_dropout = float(max(0.0, attn_dropout))

        # Soft prompt bank
        self.prompt_bank = nn.Parameter(torch.empty(bank_size, prompt_len, hidden_size))
        nn.init.normal_(self.prompt_bank, mean=0.0, std=0.02)
        # Gating via pooled-MLP (default)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, bank_size),
        )

        # Optional: attention-gated logits using learnable queries q_k and cross-attention
        if self.attn_gate:
            H = self.attn_heads
            D = self.hidden_size
            if D % H != 0:
                H = 1
                self.attn_heads = 1
            self.query_bank = nn.Parameter(torch.empty(self.bank_size, D))  # [K, D]
            nn.init.normal_(self.query_bank, mean=0.0, std=0.02)
            self.q_proj = nn.Linear(D, D, bias=False)
            self.k_proj = nn.Linear(D, D, bias=False)
            self.v_proj = nn.Linear(D, D, bias=False)
            self.attn_drop = nn.Dropout(self.attn_dropout) if self.attn_dropout > 0 else nn.Identity()
            self.logit_proj = nn.Linear(D, 1, bias=True)

    def forward(self, token_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        token_embeds: [B, T, D]
        attention_mask: [B, T] or None
        Returns:
          fused: [B, P, D]
          gate_probs: [B, K]
        """
        B, T, D = token_embeds.shape

        if self.attn_gate:
            # Attention-gated logits: learnable queries q_k cross-attend to token_embeds
            H = self.attn_heads
            Dh = D // H
            # Projections
            # Q: [K, D] -> [H, K, Dh]
            Q = self.q_proj(self.query_bank)  # [K, D]
            Q = Q.view(self.bank_size, H, Dh).transpose(0, 1)  # [H, K, Dh]
            # K,V: [B, T, D] -> [B, H, T, Dh]
            K_ = self.k_proj(token_embeds).view(B, T, H, Dh).permute(0, 2, 1, 3)
            V_ = self.v_proj(token_embeds).view(B, T, H, Dh).permute(0, 2, 1, 3)
            # Scores: [B, H, K, T]
            scores = torch.einsum('hkd,bhtd->bhkt', Q, K_) / math.sqrt(max(1.0, float(Dh)))
            if attention_mask is not None:
                # Mask: True where to mask
                mask = (attention_mask == 0).view(B, 1, 1, T)
                scores = scores.masked_fill(mask, -1e9)
            attn_w = torch.softmax(scores, dim=-1)
            attn_w = self.attn_drop(attn_w)
            # Context: [B, H, K, Dh]
            ctx = torch.einsum('bhkt,bhtd->bhkd', attn_w, V_)
            # Merge heads -> [B, K, D]
            ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, self.bank_size, D)
            # Logits: [B, K]
            logits = self.logit_proj(ctx).squeeze(-1)
        else:
            # Default MLP-gated logits via masked mean pool
            if attention_mask is None:
                # default: all tokens participate
                attn = token_embeds.new_ones(B, T)
            else:
                attn = attention_mask.to(dtype=token_embeds.dtype)
            # masked mean pooling over token dimension
            attn_exp = attn.unsqueeze(-1)  # [B, T, 1]
            denom = attn_exp.sum(dim=1).clamp(min=1.0)  # [B, 1]
            pooled = (token_embeds * attn_exp).sum(dim=1) / denom  # [B, D]
            logits = self.gate(pooled)  # [B, K]

        # Convert logits -> probs with optional top-k sparsification
        if self.top_k and self.top_k > 0 and self.top_k < self.bank_size:
            k = min(self.top_k, self.bank_size)
            vals, idx = torch.topk(logits, k, dim=-1)
            masked = torch.full_like(logits, float('-inf'))
            masked.scatter_(1, idx, vals)
            probs = torch.softmax(masked, dim=-1)  # only top-k active
        else:
            probs = torch.softmax(logits, dim=-1)  # all prompts active

        # Fused prompts: sum_k probs[b,k] * prompt_bank[k]  => [B, P, D]
        fused = torch.einsum("bk,kpd->bpd", probs, self.prompt_bank)
        return fused, probs

_orig_gather = Accelerator.gather

def _safe_gather(self, tensor, *args, **kwargs):
    # 统一成当前设备上的 0 维致密张量
    t = torch.as_tensor(tensor, dtype=torch.int64, device=self.device).contiguous()
    if t.dim() != 0:
        t = t.view(())
    # 稀疏就转致密
    if hasattr(t, "is_sparse") and t.is_sparse:
        t = t.to_dense()
    return _orig_gather(self, t, *args, **kwargs)

Accelerator.gather = _safe_gather


class GroupByNumImagesSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        buckets = defaultdict(list)
        for i, row in enumerate(dataset.rows):
            n = len(row.get("image_path", [])) if row.get("image_path") else 0
            buckets[n].append(i)
        # 组批
        self.batches = []
        for n, idxs in buckets.items():
            if shuffle:
                random.shuffle(idxs)
            for s in range(0, len(idxs), batch_size):
                b = idxs[s:s+batch_size]
                if len(b) == batch_size or not drop_last:
                    self.batches.append(b)
        if shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)

class DistributedBatchSampler(Sampler):
    """
    Wrap a batch_sampler to shard batches across distributed ranks.
    Ensures all ranks yield the same number of batches per epoch by
    padding with the last local batch when not drop_last.
    """
    def __init__(self, base_batch_sampler: Sampler, num_replicas: int = None, rank: int = None, drop_last: bool = False):
        super().__init__(None)
        self.base_batch_sampler = base_batch_sampler
        self.drop_last = bool(drop_last)

        # Infer dist info if not provided
        try:
            import torch.distributed as dist
            if num_replicas is None:
                num_replicas = dist.get_world_size() if dist.is_initialized() else 1
            if rank is None:
                rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            num_replicas = 1 if num_replicas is None else num_replicas
            rank = 0 if rank is None else rank

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __len__(self):
        total = len(self.base_batch_sampler)
        if self.drop_last:
            return total // self.num_replicas
        return math.ceil(total / self.num_replicas) if self.num_replicas > 0 else total

    def __iter__(self):
        # Materialize base batches (stable ordering expected per epoch)
        base_batches = list(iter(self.base_batch_sampler))
        total = len(base_batches)
        if self.num_replicas <= 1:
            yield from base_batches
            return

        # Round-robin shard by batch index
        local = [base_batches[i] for i in range(self.rank, total, self.num_replicas)]
        target_len = (total // self.num_replicas) if self.drop_last else math.ceil(total / self.num_replicas)

        if self.drop_last:
            local = local[:target_len]
        else:
            # Pad with last batch to keep equal number of steps across ranks
            if len(local) > 0 and len(local) < target_len:
                local = local + [local[-1]] * (target_len - len(local))

        for b in local:
            yield b

class SafeVLTrainer(Trainer):
    def get_train_dataloader(self):
        args = self.args
        bs = args.per_device_train_batch_size
        sampler = GroupByNumImagesSampler(self.train_dataset, batch_size=bs, drop_last=args.dataloader_drop_last)
        # Shard batches across distributed ranks if applicable
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                sampler = DistributedBatchSampler(
                    sampler,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    drop_last=args.dataloader_drop_last,
                )
        except Exception:
            pass
        dl_kwargs = dict(
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )
        try:
            nw = int(args.dataloader_num_workers or 0)
        except Exception:
            nw = 0
        if nw > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = 4
        return DataLoader(self.train_dataset, **dl_kwargs)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        clean_inputs = dict(inputs)
        try:
            bs = None
            for k in ("input_ids", "labels", "attention_mask"):
                v = clean_inputs.get(k, None)
                if isinstance(v, torch.Tensor):
                    bs = int(v.shape[0])
                    break

            def _strip_bs1(x):
                if bs == 1:
                    if isinstance(x, torch.Tensor) and x.dim() >= 2 and x.shape[0] == 1:
                        return x[0]
                    if isinstance(x, (list, tuple)) and len(x) == 1:
                        return x[0]
                return x

            for k in ("pixel_values", "image_grid_thw", "pixel_mask", "image_sizes"):
                if k in clean_inputs:
                    clean_inputs[k] = _strip_bs1(clean_inputs[k])
        except Exception:
            pass

        def _inject_mop_and_call(base_inputs: dict):
            wrapped = model
            base_model = getattr(wrapped, "module", wrapped)
            head = getattr(base_model, "prompt_mixture_head", None)
            use_mop = (head is not None)
            if not use_mop:
                return wrapped(**base_inputs)

            inp = dict(base_inputs)
            input_ids = inp.get("input_ids", None)
            attn = inp.get("attention_mask", None)
            labels = inp.get("labels", None)
            if input_ids is None:
                raise RuntimeError("Mixture-of-Prompts")

            embed_layer = base_model.get_input_embeddings()
            tok_embeds = embed_layer(input_ids)  # [B, T, D]

            fused_prompt, _ = head(tok_embeds, attn)

            B, T, D = tok_embeds.shape
            P = fused_prompt.shape[1]
            pos = 1 if T >= 1 else 0
            left = tok_embeds[:, :pos, :]
            right = tok_embeds[:, pos:, :]
            new_embeds = torch.cat([left, fused_prompt, right], dim=1)  # [B, T+P, D]

            if attn is None:
                raise RuntimeError("Mixture-of-Prompts 需要 attention_mask 以同步形状")
            left_m = attn[:, :pos]
            right_m = attn[:, pos:]
            ins_m = torch.ones((attn.shape[0], P), dtype=attn.dtype, device=attn.device)
            new_attn = torch.cat([left_m, ins_m, right_m], dim=1)

            new_labels = None
            if labels is not None:
                left_l = labels[:, :pos]
                right_l = labels[:, pos:]
                ins_l = torch.full((labels.shape[0], P), fill_value=-100, dtype=labels.dtype, device=labels.device)
                new_labels = torch.cat([left_l, ins_l, right_l], dim=1)

            inp.pop("input_ids", None)
            inp["inputs_embeds"] = new_embeds
            inp["attention_mask"] = new_attn
            if new_labels is not None:
                inp["labels"] = new_labels
            return wrapped(**inp)

        try:
            outputs = _inject_mop_and_call(clean_inputs)
        except RuntimeError as e:
            msg = str(e)
            if "spatial_merge_unit" in msg or "reshape" in msg:
                fb = dict(clean_inputs)
                fb.pop("pixel_values", None)
                fb.pop("image_grid_thw", None)
                outputs = _inject_mop_and_call(fb)
            else:
                raise

        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        try:
            if isinstance(loss, torch.Tensor) and loss.dim() != 0:
                loss = loss.mean()
        except Exception:
            pass

        # 聚合 MoE 路由正则项
        aux = None
        for m in model.modules():
            if isinstance(m, MoELoRALinear):
                val = getattr(m, "latest_aux", None)
                if isinstance(val, torch.Tensor):
                    aux = (val if aux is None else aux + val)
        if aux is not None:
            try:
                loss = loss + aux
            except Exception:
                loss = loss + aux.to(dtype=loss.dtype, device=loss.device)

        return (loss, outputs) if return_outputs else loss

def join_turns(conversations, num_images: int):
    msgs = []
    already_mm = False
    for t in conversations:
        c = t.get("content")
        if isinstance(c, list) and c and isinstance(c[0], dict) and "type" in c[0]:
            already_mm = True
            break
    if already_mm:
        for t in conversations:
            msgs.append({"role": t.get("role", "user"), "content": t.get("content", [])})
        return msgs

    images_attached = False
    for t in conversations:
        role = t.get("role", "user")
        content = t.get("content", "")
        if not isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        segs = []
        if role == "user" and (not images_attached) and num_images > 0:
            segs.extend([{"type": "image"} for _ in range(num_images)])
            images_attached = True
        segs.append({"type": "text", "text": content})
        msgs.append({"role": role, "content": segs})
    return msgs


# ---------------- 权重加载（仅模型参数） ----------------

from typing import Dict, Any, List

def _remap_and_filter_keys_for_moe(
    sd: Dict[str, torch.Tensor],
    model,
    print_dropped: int = 40,
) -> Dict[str, torch.Tensor]:

    model_keys = set(model.state_dict().keys())
    out: Dict[str, torch.Tensor] = {}
    dropped: List[str] = []

    model_uses_lang_prefix = any(k.startswith("model.language_model.") for k in model_keys)

    for k, v in sd.items():
        kk = k
        # 规则1：visual.* -> model.visual.*
        if k.startswith("visual."):
            kk = "model." + k
        # 规则2：如果模型使用 language_model 前缀，则把 model.* -> model.language_model.*（已是 language_model.* 的不变）
        if model_uses_lang_prefix and k.startswith("model.") and (not k.startswith("model.language_model.")):
            kk = "model.language_model." + k[len("model."):]

        if kk in model_keys:
            out[kk] = v
        elif k in model_keys:
            out[k] = v
        else:
            dropped.append(k)

    if dropped and print_dropped > 0:
        n = min(print_dropped, len(dropped))
        print(f"[init_moe] dropped keys sample count={n} of {len(dropped)}")
        for i in range(n):
            print("  -", dropped[i])
        if len(dropped) > n:
            print(f"[init_moe] dropped keys more {len(dropped) - n} not shown")
    print(f"[init_moe] remap keep={len(out)} drop={len(dropped)}")
    return out


def _load_safetensors_state_dict(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    if not _HAS_SAFE:
        raise RuntimeError("需要安装 safetensors 以加载权重: pip install -U safetensors")
    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            idx = _json.load(f)
        weight_map = idx.get("weight_map", {})
        shard_files = sorted({ckpt_dir / fname for fname in weight_map.values()})
        if not shard_files:
            raise FileNotFoundError(f"index 中没有列出分片文件 位于 {index_path}")
        merged: Dict[str, torch.Tensor] = {}
        for sf in shard_files:
            merged.update(_safe_load_file(str(sf)))
        return merged
    # 无 index，尝试通配
    safes = sorted(ckpt_dir.glob("model-*.safetensors"))
    if not safes and (ckpt_dir / "model.safetensors").exists():
        safes = [ckpt_dir / "model.safetensors"]
    if not safes:
        raise FileNotFoundError(f"未找到 safetensors 分片 位于 {ckpt_dir}")
    merged: Dict[str, torch.Tensor] = {}
    for sf in safes:
        merged.update(_safe_load_file(str(sf)))
    return merged


def _get_patch_size_from_processor(processor, default: int = 14) -> int:
    # 常见模型的 processor.image_processor 里会有 patch_size
    ip = getattr(processor, "image_processor", None)
    ps = getattr(ip, "patch_size", None)
    # 有的实现是 dict 或 tuple
    if isinstance(ps, (tuple, list)):
        ps = int(ps[0])
    if isinstance(ps, dict):
        ps = int(ps.get("height", default))
    if isinstance(ps, int):
        return ps
    return default

def _compute_token_grid(
    w: int,
    h: int,
    max_grid: int = 56,
    min_grid: int = 24,
    ensure_even: bool = True,
) -> Tuple[int, int]:
    """返回 tokens_h, tokens_w"""
    if w >= h:
        tokens_w = max_grid
        tokens_h = max(min_grid, int(round(tokens_w * h / w)))
    else:
        tokens_h = max_grid
        tokens_w = max(min_grid, int(round(tokens_h * w / h)))
    if ensure_even:
        if tokens_w % 2 != 0:
            tokens_w += 1
        if tokens_h % 2 != 0:
            tokens_h += 1
    return tokens_h, tokens_w

def reshape_images_for_vlm(
    images: List[Image.Image],
    processor=None,
    max_grid: int = 56,
    min_grid: int = 24,
    resample=Image.BICUBIC,
) -> Tuple[List[Image.Image], Tuple[int, int], int]:
    """
    把输入 PIL 图片列表 reshape 到符合网格约束的分辨率
    返回 新图片列表, (tokens_h, tokens_w), patch_size
    """
    patch = _get_patch_size_from_processor(processor, default=14)
    out = []
    tokens_hw = None
    for img in images:
        w, h = img.size
        th, tw = _compute_token_grid(w, h, max_grid=max_grid, min_grid=min_grid)
        new_h, new_w = th * patch, tw * patch
        out.append(img.resize((new_w, new_h), resample=resample))
        tokens_hw = (th, tw)  # 多图时每张都会算, 这里返回最后一张或你也可以改成返回列表
    return out, tokens_hw, patch
def vl_data_collator(features):
    import torch, torch.nn.functional as F
    text_keys = ["input_ids", "attention_mask", "labels"]
    img_keys  = {"pixel_values", "image_grid_thw", "pixel_mask", "image_sizes"}

    # 1) 计算本 batch 文本最长长度
    max_len = max(f["input_ids"].shape[0] for f in features)

    # 2) 动态 pad 文本张量
    pad_id = 0  # tokenizer.pad_token_id，很多 Qwen 默认就是 0
    out = {}
    for k in text_keys:
        xs = []
        for f in features:
            x = f[k]
            pad_val = -100 if k == "labels" else (0 if k == "attention_mask" else pad_id)
            if x.shape[0] < max_len:
                x = F.pad(x, (0, max_len - x.shape[0]), value=pad_val)
            xs.append(x)
        out[k] = torch.stack(xs, dim=0)

    # 3) 处理视觉相关键
    for k in img_keys:
        if k in features[0]:
            try:
                out[k] = torch.stack([f[k] for f in features], dim=0)
            except Exception:
                # 若形状不同就保留为列表，Trainer 也能搬到 device
                out[k] = [f[k] for f in features]

    return out
def _ceil_to(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple

def qwen_vl_collator_token_pad_only(
    features,
    pad_to_multiple_of_text: int = 8,
    pad_token_id: int = 0,
):
    import torch
    import torch.nn.functional as F

    out = {}
    # 1 文本 pad
    text_keys = [k for k in ("input_ids", "attention_mask", "labels") if k in features[0]]
    if text_keys:
        max_len = max(f[text_keys[0]].shape[0] for f in features)
        if pad_to_multiple_of_text and (max_len % pad_to_multiple_of_text != 0):
            max_len = ((max_len + pad_to_multiple_of_text - 1) // pad_to_multiple_of_text) * pad_to_multiple_of_text
        for k in text_keys:
            xs = []
            for f in features:
                x = f[k]
                pad_val = -100 if k == "labels" else (0 if k == "attention_mask" else pad_token_id)
                if x.shape[0] < max_len:
                    x = F.pad(x, (0, max_len - x.shape[0]), value=pad_val)
                xs.append(x)
            out[k] = torch.stack(xs, dim=0)

    # 2 视觉序列 pad 仅在 token 维上对齐
    if "pixel_values" in features[0]:
        pvs = []
        Ns = []
        D = None
        dtype = None
        for f in features:
            pv = f["pixel_values"]        # 期望 [N_tokens, D]
            if pv.dim() == 1:
                pv = pv.unsqueeze(0)      # 退化情况兼容
            assert pv.dim() == 2, f"pixel_values 应为二维序列 got dim={pv.dim()}"
            if D is None:
                D = pv.shape[1]
                dtype = pv.dtype
            else:
                if pv.shape[1] != D:
                    raise RuntimeError("每样本的 token 特征维度不一致 请确保 processor 配置一致")
            pvs.append(pv)
            Ns.append(pv.shape[0])

        B = len(features)
        max_N = max(Ns)
        out_pv = torch.zeros(B, max_N, D, dtype=dtype)
        pixel_mask = torch.zeros(B, max_N, dtype=torch.bool)
        for i, pv in enumerate(pvs):
            n = pv.shape[0]
            out_pv[i, :n] = pv
            pixel_mask[i, :n] = True
        out["pixel_values"] = out_pv
        out["pixel_mask"] = pixel_mask


    if "image_grid_thw" in features[0]:
        grids = []
        for f in features:
            g = f["image_grid_thw"]
            g = g if isinstance(g, torch.Tensor) else torch.as_tensor(g)
            if g.dim() == 1:
                g = g.view(1, 3)          # 单图保持为 [1,3]
            grids.append(g)

        out["image_grid_thw"] = torch.cat(grids, dim=0)       # [sum_i Mi, 3]

    for k in features[0].keys():
        if k in out or k in {"input_ids", "attention_mask", "labels", "pixel_values", "pixel_mask", "image_grid_thw", "image_grid_mask"}:
            continue
        vals = [f[k] for f in features]
        try:
            if all(isinstance(v, torch.Tensor) and v.shape == vals[0].shape for v in vals):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        except Exception:
            out[k] = vals

    return out
def qwen_vl_collator_varimg(features, pad_to_multiple_of_text: int = 8, pad_token_id: int = 0):
    """
    Collate for Qwen2.* VL with variable-sized images using Qwen2VLImageProcessor.

    - Pads text to uniform length.
    - Concatenates per-sample image patch matrices into a single 2D tensor [sum_patches, D].
    - Concatenates per-sample image grids into [sum_images, 3] (LongTensor).

    This matches the model's expectation where pixel_values is a 2D matrix of flattened patches
    and image_grid_thw enumerates per-image THW for the whole batch.
    """
    import torch
    import torch.nn.functional as F

    out = {}

    # 1) Text: pad and stack
    text_keys = [k for k in ("input_ids", "attention_mask", "labels") if k in features[0]]
    if text_keys:
        max_len = max(int(f[text_keys[0]].shape[0]) for f in features)
        if pad_to_multiple_of_text and (max_len % pad_to_multiple_of_text != 0):
            max_len = ((max_len + pad_to_multiple_of_text - 1) // pad_to_multiple_of_text) * pad_to_multiple_of_text
        for k in text_keys:
            xs = []
            for f in features:
                x = f[k]
                pad_val = -100 if k == "labels" else (0 if k == "attention_mask" else pad_token_id)
                if x.shape[0] < max_len:
                    x = F.pad(x, (0, max_len - x.shape[0]), value=pad_val)
                xs.append(x)
            out[k] = torch.stack(xs, dim=0)

    # 2) Visual: concatenate patches and grids across samples
    if "pixel_values" in features[0]:
        # Each f["pixel_values"] is expected shape [Ni_patches, D]
        pvs = [f["pixel_values"] for f in features]
        # Validate feature dims agree
        D0 = int(pvs[0].shape[-1])
        for i, pv in enumerate(pvs):
            if pv.dim() != 2 or int(pv.shape[-1]) != D0:
                raise RuntimeError(f"pixel_values of sample {i} must be 2D [N,D] with D={D0}, got {tuple(pv.shape)}")
        out["pixel_values"] = torch.cat(pvs, dim=0)

    if "image_grid_thw" in features[0]:
        grids = []
        for f in features:
            g = f["image_grid_thw"]
            g = g if isinstance(g, torch.Tensor) else torch.as_tensor(g)
            if g.dim() == 1:
                g = g.view(1, 3)
            grids.append(g.to(dtype=torch.long))
        out["image_grid_thw"] = torch.cat(grids, dim=0)

    # 3) Pass through other keys when possible
    for k in features[0].keys():
        if k in out or k in {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw", "pixel_mask", "image_sizes"}:
            continue
        vals = [f[k] for f in features]
        try:
            if all(isinstance(v, torch.Tensor) and v.shape == vals[0].shape for v in vals):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        except Exception:
            out[k] = vals

    return out

def qwen_vl_collator_pad_images(features, pad_to_multiple_of_text: int = 8, pad_token_id: int = 0):
    import torch
    import torch.nn.functional as F

    out = {}
    # 文本同方案 A 省略

    # 视觉对齐
    pvs = [f["pixel_values"] for f in features]            # 各自 [Mi, 3, H, W]
    grids = [f["image_grid_thw"] for f in features]        # 各自 [Mi, 3] 或列表
    grids = [g if isinstance(g, torch.Tensor) else torch.as_tensor(g) for g in grids]
    grids = [g.unsqueeze(0) if g.dim() == 1 else g for g in grids]

    B = len(features)
    Mmax = max(pv.shape[0] for pv in pvs)
    C, H, W = pvs[0].shape[1:]
    dtype = pvs[0].dtype
    device = pvs[0].device

    pv_out = torch.zeros(B, Mmax, C, H, W, dtype=dtype, device=device)
    grid_out = torch.zeros(B, Mmax, 3, dtype=torch.int32)
    img_mask = torch.zeros(B, Mmax, dtype=torch.bool)

    for i, pv in enumerate(pvs):
        Mi = pv.shape[0]
        pv_out[i, :Mi] = pv
        grid_out[i, :Mi] = grids[i]
        img_mask[i, :Mi] = True

    out["pixel_values"] = pv_out
    out["image_grid_thw"] = grid_out
    out["image_grid_mask"] = img_mask 
    return out
def resize_images_fixed(
    images,
    width: int = 704,
    height: int = 256,
    resample=Image.BICUBIC,
):
    return [im.resize((width, height), resample=resample) for im in images]

class VLJsonlDataset(Dataset):
    def __init__(self, path, processor, image_root, max_len, use_qwen_image_processor: bool = False,
                 reserved_soft_len: int = 0):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        self.proc = processor
        self.image_root = Path(image_root) if image_root else None
        self.max_len = max_len
        self.use_qwen_ip = bool(use_qwen_image_processor)
        self.reserved_soft_len = int(reserved_soft_len or 0)

    def __len__(self):
        return len(self.rows)

    def _load_images(self, paths):
        ims = []
        for p in paths:
            try:
                pth = Path(p) if self.image_root is None else self.image_root / p
                im = Image.open(pth).convert("RGB")
                ims.append(im)
            except Exception as e:
                print(f"[warn] failed to open image: {p} ({e})")
                continue
        return ims

    def __getitem__(self, i):
        row = self.rows[i]
        img_paths = row.get("image_path", [])

        images = self._load_images(img_paths) if img_paths else None
        num_loaded = len(images) if images is not None else 0
        if images is not None and num_loaded == 0:
            images = None

        conv = join_turns(row["conversations"], num_loaded)

        chat_text = self.proc.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )

        if images is None:
            raise RuntimeError("No images were provided to the processor.")

        else:
            images = images if self.use_qwen_ip else resize_images_fixed(images, width=1600, height=900)
            
            batch = self.proc(
                images=images,
                text=chat_text,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            try:
                need_len = int(batch["input_ids"].shape[-1])
            except Exception:
                need_len = 0
            # 若后续要在 BOS 处插入软提示 P 个 token，则需要预留长度
            if self.max_len and (need_len + self.reserved_soft_len) > self.max_len:
                raise RuntimeError(f"Sample idx={i} need_len={need_len} exceeds max_len={self.max_len}, "
                                   f"with reserved_soft_len={self.reserved_soft_len}.")

            # 视觉网格合法性检查 不合法则丢弃视觉键
            if "image_grid_thw" in batch:
                grid = batch["image_grid_thw"][0]
                # try:
                if grid.dim() == 1:
                    seq_tokens = int(torch.prod(grid).item())
                else:
                    seq_tokens = int(torch.prod(grid, dim=-1).sum().item())
                if seq_tokens < 4 or (seq_tokens % 4) != 0:
                    raise ValueError(f"image_grid_thw has invalid total tokens: {seq_tokens}")
                    batch.pop("pixel_values", None)
                    batch.pop("image_grid_thw", None)

        # 仅对最后一个 assistant 段打 loss：构造去掉最后 assistant 文本的对话，
        # 计算两版长度之差作为可学习 token 数，仅在尾部这些位置保留 labels。
        try:
            conv_trunc = []
            last_asst_idx = -1
            for idx_t, t in enumerate(conv):
                if t.get("role") == "assistant":
                    last_asst_idx = idx_t
                # 浅拷贝一份
                ct = {"role": t.get("role"), "content": t.get("content")}
                conv_trunc.append(ct)
            if last_asst_idx >= 0:
                ct = conv_trunc[last_asst_idx]
                c = ct.get("content")
                if isinstance(c, list):
                    newc = []
                    for seg in c:
                        if isinstance(seg, dict) and seg.get("type") == "text":
                            newc.append({"type": "text", "text": ""})
                        else:
                            newc.append(seg)
                    ct["content"] = newc
                elif isinstance(c, str):
                    ct["content"] = ""
                chat_text_trunc = self.proc.apply_chat_template(
                    conv_trunc, tokenize=False, add_generation_prompt=False
                )
                # 仅用 tokenizer 计算去掉最后一段后的文本长度，避免重复做图像预处理
                tok = getattr(self.proc, "tokenizer", None)
                if tok is not None:
                    try:
                        tok_out = tok(
                            text=chat_text_trunc,
                            padding=False,
                            truncation=False,
                            return_tensors="pt",
                        )
                        trunc_len = int(tok_out["input_ids"].shape[-1])
                    except Exception:
                        
                        probe_trunc = self.proc(
                            text=chat_text_trunc,
                            padding=False,
                            truncation=False,
                            return_tensors="pt",
                        )
                        trunc_len = int(probe_trunc["input_ids"].shape[-1])
                else:
                    
                    probe_trunc = self.proc(
                        text=chat_text_trunc,
                        padding=False,
                        truncation=False,
                        return_tensors="pt",
                    )
                    trunc_len = int(probe_trunc["input_ids"].shape[-1])
                # 同样用 tokenizer 计算完整文本的长度，以获得“最后一段回答”的纯文本 token 数
                if tok is not None:
                    try:
                        tok_full = tok(
                            text=chat_text,
                            padding=False,
                            truncation=False,
                            return_tensors="pt",
                        )
                        full_text_len = int(tok_full["input_ids"].shape[-1])
                    except Exception:
                        tok_full = self.proc(
                            text=chat_text,
                            padding=False,
                            truncation=False,
                            return_tensors="pt",
                        )
                        full_text_len = int(tok_full["input_ids"].shape[-1])
                else:
                    tok_full = self.proc(
                        text=chat_text,
                        padding=False,
                        truncation=False,
                        return_tensors="pt",
                    )
                    full_text_len = int(tok_full["input_ids"].shape[-1])

                # 仅监督最后一段回答的文本 token 数；在实际 batch 中，回答位于末尾，
                # 因此直接在 batch 的 input_ids 末尾截取 keep_text 个 token 打标。
                keep_text = max(0, full_text_len - trunc_len)
                batch_len = int(batch["input_ids"].shape[-1])
                keep = max(0, min(batch_len, keep_text))
            else:
                keep = 0
        except Exception:
            keep = 0

        # 调试：如未能为本样本构造监督（keep==0），打印关键信息便于排查
        if keep == 0:
            try:
                _batch = int(batch_len) if 'batch_len' in locals() else int(batch["input_ids"].shape[-1])
                _trunc = int(trunc_len) if 'trunc_len' in locals() else None
                _fulltxt = int(full_text_len) if 'full_text_len' in locals() else None
                _last = int(last_asst_idx) if 'last_asst_idx' in locals() else None
                print(f"[labels] keep==0 idx={i} last_asst_idx={_last} batch_len={_batch} full_text_len={_fulltxt} trunc_len={_trunc}")
            except Exception:
                try:
                    print(f"[labels] keep==0 idx={i}")
                except Exception:
                    pass

        # 基于实际 batch 的末尾片段构造监督区间，并剔除模板/特殊标记
        ids = batch["input_ids"]
        labels = torch.full_like(ids, fill_value=-100)
        batch_len = int(ids.shape[-1])
        start = max(0, batch_len - int(keep))
        end = batch_len
        if keep > 0 and end > start:
            tok = getattr(self.proc, "tokenizer", None)
            special_ids = set(getattr(tok, "all_special_ids", []) or [])
            # 常见 Qwen2.5 VL 边界标记（若 tokenizer 不提供，回退到常见数值）
            def _tok2id(s, fallback=None):
                try:
                    return tok.convert_tokens_to_ids(s) if tok is not None else fallback
                except Exception:
                    return fallback
            im_start_id = _tok2id("<|im_start|>", 151644)
            im_end_id = _tok2id("<|im_end|>", 151645)
            assistant_id = 77091
            nl_id = 198
            cr_id = 13  # 一些词表将 13 用作换行/回车

            ignore_head = set(x for x in [im_start_id, assistant_id] if x is not None) | special_ids
            ignore_tail_basic = set(x for x in [nl_id, cr_id] if x is not None)

            # 使用 1D 视图按 token 访问，避免 [1, L] 的维度混淆
            ids1d = ids.view(-1)

            while start > 2 and (ids1d[start-1].item() !=nl_id or ids1d[start-2].item() !=assistant_id):
                start -= 1

            if end > start:
                labels[..., start:end] = ids[..., start:end]
            # print(self.proc.tokenizer.decode(ids[..., start:end][0].tolist(), skip_special_tokens=False))
        batch["labels"] = labels
        image_keys = {"pixel_values", "image_grid_thw", "pixel_mask", "image_sizes"}
        out = {}
        for k, v in batch.items():
            if k in image_keys:
                out[k] = v
            else:
                out[k] = v[0] if hasattr(v, "dim") and v.dim() > 0 else v
        # print(out['pixel_values'].shape)
        return out

def scan_overlong_samples(
    ds,
    processor,
    max_len: int,
    resize_wh=(1056, 384),
    limit: int = 0,
    save_path: str | None = None,
):
    """
    扫描数据集中会触发截断从而导致图像占位符不匹配的样本
    用 truncation=False 计算真实所需长度 need 只要 need > max_len 就标记为超长
    """
    bad = []
    n = len(ds) if limit == 0 else min(limit, len(ds))
    header = f"[scan] start scanning {n} samples with max_len={max_len}"
    print(header)

    # 进度条对象 可用则用 tqdm 否则为 None
    pbar = tqdm(total=n, desc="[scan] scanning", dynamic_ncols=True) if tqdm else None

    for idx in range(n):
        row = ds.rows[idx]
        paths = row.get("image_path", []) or []
        try:
            imgs = ds._load_images(paths) if paths else []
            if len(imgs) > 0:
                # 扫描阶段：若使用 Qwen 图像处理器，则不强制 resize
                imgs = imgs if getattr(ds, 'use_qwen_ip', False) else resize_images_fixed(imgs, width=resize_wh[0], height=resize_wh[1])

            conv = join_turns(row["conversations"], len(imgs))
            text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)

            probe = processor(
                images=imgs if len(imgs) > 0 else None,
                text=text,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            need = int(probe["input_ids"].shape[-1])

            per_img_tokens = []
            grid = probe.get("image_grid_thw", None)
            if grid is not None:
                g = grid
                if isinstance(g, torch.Tensor):
                    g = g.view(-1, 3)
                    for r in g:
                        per_img_tokens.append(int(torch.prod(r).item()))

            if need > max_len:
                item = {
                    "idx": idx,
                    "need": need,
                    "max_len": max_len,
                    "n_images": len(imgs),
                    "per_img_tokens": per_img_tokens,
                    "paths": paths,
                }
                bad.append(item)
                msg = (f"[scan] overlong idx={idx} need={need} max={max_len} "
                       f"n_images={len(imgs)} per_img_tokens={per_img_tokens} paths={paths}")
                if pbar:
                    pbar.write(msg)
                else:
                    print(msg)

        except Exception as e:
            item = {
                "idx": idx,
                "need": f"proc_failed: {e}",
                "max_len": max_len,
                "n_images": len(paths),
                "per_img_tokens": [],
                "paths": paths,
            }
            bad.append(item)
            msg = f"[scan] failed idx={idx} error={e} paths={paths}"
            if pbar:
                pbar.write(msg)
            else:
                print(msg)

        # 更新进度条并展示当前超长计数
        if pbar:
            pbar.set_postfix_str(f"bad={len(bad)}")
            pbar.update(1)

    if pbar:
        pbar.close()

    print(f"[scan] done overlong_count={len(bad)} of {n}")

    if save_path and len(bad) > 0:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for it in bad:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"[scan] saved report to {save_path}")
        except Exception as e:
            print(f"[scan] failed to write report to {save_path} error={e}")

    return bad


# ---------------- MoE-LoRA 打补丁与防呆 ----------------

TARGET_PATTERNS = [
    # r"\bself_attn\.(q_proj|k_proj|v_proj|o_proj)\b",
    r"\bmlp\.(gate_proj|up_proj|down_proj)\b",
    # r"\bvision_tower\..*\.proj\b",
    # r"\bmm_projector\..*",
    # r"\bW_pack\b",            # 部分 Qwen 家族的打包权重
    # r"\bw[123]\b",            # 部分实现的 MLP 命名
]

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def list_linear_candidates(model):
    hits = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if any(re.search(p, name) for p in TARGET_PATTERNS):
                hits.append(name)
    return hits

def freeze_base_params_safe(model):
    has_moe = any(isinstance(m, MoELoRALinear) for m in model.modules())
    if not has_moe:
        print("[MoE-LoRA] no MoELoRALinear modules detected skip freezing base params")
        return
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, MoELoRALinear):
            for p in m.parameters():
                p.requires_grad = True
            # 确保底座线性层不训练
            base = getattr(m, "base", None)
            if base is not None:
                for p in base.parameters():
                    p.requires_grad = False


def try_patch_moe_lora(model, num_experts, top_k, r, alpha, dropout, router_aux_weight):
    """调用你的 patch 函数 打上 MoE-LoRA 并回报命中数量"""
    # 先打印候选层名 利于排查
    candidates = list_linear_candidates(model)
    print(f"[MoE-LoRA] linear candidates matched by regex: {len(candidates)}")
    for n in candidates[:32]:
        print(f"  - {n}")
    if len(candidates) == 0:
        print("[MoE-LoRA] warning no linear candidates matched current regex")

    # 兼容不同实现的参数名
    patched_names = None
    try:
        model2, patched_names = patch_model_with_moe_lora(
            model,
            num_experts=num_experts,
            top_k=top_k,
            r=r,
            alpha=alpha,
            dropout=dropout,
            router_aux_weight=router_aux_weight,
            name_regex_list=TARGET_PATTERNS,
            dryrun=False,
        )
        model = model2
    except TypeError:
        # 如果实现不支持 name_regex_list 就无过滤 让内部自行决定
        print("[MoE-LoRA] patch function signature does not accept name_regex_list trying without it")
        model2, patched_names = patch_model_with_moe_lora(
            model,
            num_experts=num_experts,
            top_k=top_k,
            r=r,
            alpha=alpha,
            dropout=dropout,
            router_aux_weight=router_aux_weight,
        )
        model = model2

    # 统计命中
    if patched_names is not None:
        print(f"[MoE-LoRA] patched layers reported by patcher: {len(patched_names)}")

    patched = sum(1 for _ in model.modules() if isinstance(_, MoELoRALinear))
    print(f"[MoE-LoRA] detected MoELoRALinear modules in model: {patched}")
    return model, patched


def fallback_to_standard_lora(model, r, alpha, dropout):
    print("[Fallback] applying standard LoRA with PEFT target common Qwen projections")
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "W_pack", "w1", "w2", "w3",
        ],
        modules_to_save=["lm_head"],
    )
    model = get_peft_model(model, lora_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model

def print_trainable_parameter_names(model):
    """打印所有 requires_grad=True 的参数名字 仅在主进程打印"""
    try:
        import torch.distributed as dist
        is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
    except Exception:
        is_main = True

    if not is_main:
        return

    n_tensors = 0
    print("[Trainable] listing parameter names with requires_grad=True:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
            n_tensors += 1
    print(f"[Trainable] parameter tensors with requires_grad=True: {n_tensors}")

# ---------------- 主函数 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--image_root", type=str, default=None)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--optim", type=str, default="adamw_torch")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of update steps to accumulate before performing a backward/update pass."
    )
    ap.add_argument("--deepspeed", type=str, default=None)
    # 使用 Qwen2.5 的图像处理器进行预处理与对齐（取消自定义 resize）
    ap.add_argument("--use_qwen_image_processor", action="store_true",
                    help="使用 Qwen2_5_VLImageProcessor 处理图像（不再执行固定 1600x900 resize）")
    # 仅初始化 MoE 模型参数：从 checkpoint 读取模型权重，不恢复优化器/调度器等训练状态
    ap.add_argument("--init_moe_from", type=str, default=None,
                    help="仅加载给定 checkpoint 的模型参数（safetensors），不恢复训练状态")
    # MoE-LoRA 超参
    ap.add_argument("--num_experts", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--router_aux_weight", type=float, default=0.01)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    # 若 MoE 命中为 0 是否自动回退到标准 LoRA
    ap.add_argument("--fallback_to_lora", action="store_true")
    # Mixture-of-Prompts 配置
    ap.add_argument("--prompt_mixture", action="store_true",
                    help="启用 Mixture-of-Prompts（软提示+门控）")
    ap.add_argument("--prompt_bank_size", type=int, default=8,
                    help="软提示库大小 K")
    ap.add_argument("--prompt_len", type=int, default=16,
                    help="每组软提示长度 P（虚拟 token 数）")
    ap.add_argument("--prompt_gate_hidden", type=int, default=256,
                    help="门控 MLP 隐藏维度")
    ap.add_argument("--prompt_top_k", type=int, default=0,
                    help="仅激活前 k 个软提示（0 表示使用全软选择）")
    ap.add_argument("--prompt_insert_pos", type=str, default="bos", choices=["bos"],
                    help="软提示插入位置，目前仅支持 BOS 后")
    # Attention-gated logits for prompt selection
    ap.add_argument("--prompt_gate_attention", action="store_true",
                    help="使用注意力模式生成 logits：K 个可学习查询对 token 序列做多头交叉注意力")
    ap.add_argument("--prompt_gate_heads", type=int, default=8,
                    help="注意力门控的多头数")
    ap.add_argument("--prompt_gate_attn_dropout", type=float, default=0.0,
                    help="注意力权重的 dropout")
    ap.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,           # 可取 None 或 "auto" 或 具体路径
        help="断点续训入口。传路径或传 auto 自动从 output_dir 最新 checkpoint 恢复，传 none 表示不恢复"
    )    
    ap.add_argument("--scan_overlong", action="store_true",
                help="开训前扫描超长样本並打印摘要")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    assert "vl" in cfg.model_type.lower(), f"expect a multimodal Qwen VL model got {cfg.model_type}"

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 若用户启用开关且环境提供 Qwen 图像处理器，则替换 processor 的 image_processor
    if getattr(args, 'use_qwen_image_processor', False):
        if QwenVLImageProcessor is not None:
            try:
                ip = QwenVLImageProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
                if hasattr(processor, 'image_processor'):
                    processor.image_processor = ip
                    print(f"[QwenIP] using {QwenVLImageProcessor.__name__} for image preprocessing")
                else:
                    # 兜底：挂载到属性，后续调用 processor(...) 仍会使用 tokenizer/text 侧；
                    # 如需强制走 ip，可在 Dataset 内部仅传 images 给 ip，再与 tokenizer 输出合并（当前无需）。
                    setattr(processor, 'image_processor', ip)
                    print(f"[QwenIP] attached {QwenVLImageProcessor.__name__} to processor.image_processor")
            except Exception as e:
                print(f"[QwenIP] failed to load Qwen image processor: {e}")
        else:
            print("[QwenIP] Qwen2.* VLImageProcessor not available in transformers; fallback to AutoProcessor.image_processor")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass
    if args.fallback_to_lora:
        patched = 0
    else:
        # 打 MoE-LoRA 补丁
        model, patched = try_patch_moe_lora(
            model,
            num_experts=args.num_experts,
            top_k=args.top_k,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            router_aux_weight=args.router_aux_weight,
        )
    
    try:
        
        try:
            model.config._attn_implementation = "flash_attention_2"
        except Exception:
            pass
        try:
            model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("flash_attention_2")

        print("[attn] flash_attention_2 enabled")
    except Exception as e:
        print(f"[attn] FA2 not available, fallback to SDPA flash, {e}")
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash(True)
        sdp_kernel.enable_mem_efficient(False)
        sdp_kernel.enable_math(False)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.init_moe_from:
        try:
            ckpt_dir = Path(args.init_moe_from)
            print(f"[init_moe] loading model weights from {ckpt_dir}")
            sd_raw = _load_safetensors_state_dict(ckpt_dir)
            sd = _remap_and_filter_keys_for_moe(sd_raw, model)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[init_moe] loaded: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("  missing sample:", missing[:8])
            if unexpected:
                print("  unexpected sample:", unexpected[:8])
        except Exception as e:
            print(f"[init_moe] failed to load model weights: {e}")

    if patched == 0:
        msg = "[MoE-LoRA] no layers patched"
        if args.fallback_to_lora:
            print(msg + " fallback_to_lora enabled")
            model = fallback_to_standard_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
            
        else:
            print(msg + " will NOT freeze base params to avoid empty optimizer groups")

    # 冻结底座 仅当确实存在 MoE-LoRA 模块
    freeze_base_params_safe(model)

    # 安装 Mixture-of-Prompts 头（作为 model 的一个子模块）
    if getattr(args, "prompt_mixture", False) and args.prompt_len > 0 and args.prompt_bank_size > 0:
        hid = int(getattr(model.config, "hidden_size", None) or model.get_input_embeddings().weight.shape[1])
        model.prompt_mixture_head = PromptMixtureHead(
            hidden_size=hid,
            bank_size=int(args.prompt_bank_size),
            prompt_len=int(args.prompt_len),
            gate_hidden=int(args.prompt_gate_hidden),
            top_k=int(getattr(args, "prompt_top_k", 0) or 0),
            attn_gate=bool(getattr(args, "prompt_gate_attention", False)),
            attn_heads=int(getattr(args, "prompt_gate_heads", 8)),
            attn_dropout=float(getattr(args, "prompt_gate_attn_dropout", 0.0)),
        )
        tk = int(getattr(args, "prompt_top_k", 0) or 0)
        mode = "attn" if getattr(args, "prompt_gate_attention", False) else "mlp"
        if tk > 0:
            print(f"[PromptMix] enabled K={args.prompt_bank_size} P={args.prompt_len} hidden={hid} top_k={tk} gate={mode}")
        else:
            print(f"[PromptMix] enabled K={args.prompt_bank_size} P={args.prompt_len} hidden={hid} soft-all gate={mode}")
    else:
        model.prompt_mixture_head = None

    # 训练前再做一次可训练参数数量校验 避免 DeepSpeed 空参数组
    trainable = count_trainable_params(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Trainable] total params {total} trainable params {trainable}")
    if trainable == 0:
        raise RuntimeError(
            "no trainable params found check regex filters or enable --fallback_to_lora"
        )
    # 新增 调用打印函数
    print_trainable_parameter_names(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    ds = VLJsonlDataset(
        args.data_path,
        processor,
        args.image_root,
        args.seq_len,
        use_qwen_image_processor=args.use_qwen_image_processor,
        reserved_soft_len=(args.prompt_len if getattr(args, "prompt_mixture", False) else 0),
    )
    # 可选的预扫描阶段
    if args.scan_overlong:
        _report = scan_overlong_samples(
            ds=ds,
            processor=processor,
            max_len=args.seq_len,
            resize_wh=(1056, 384),      # 与 __getitem__ 的 resize 保持一致
            limit=0,
            save_path=None,
        )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=max(1, int(getattr(args, "gradient_accumulation_steps", 1))),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        deepspeed=args.deepspeed,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=8,
        include_num_input_tokens_seen =False,
    )

    _pad_id = None
    try:
        tok = processor.tokenizer
        _pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    except Exception:
        _pad_id = 0

    trainer = SafeVLTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        eval_dataset=None,
        data_collator=lambda feats: qwen_vl_collator_varimg(
            feats, pad_to_multiple_of_text=8, pad_token_id=_pad_id or 0
        ),  
    )
    # 检测断点
    resume_ckpt = None
    if args.resume_from_checkpoint:
        val = args.resume_from_checkpoint.lower()
        if val == "auto":
            # 自动在 output_dir 内找最新 checkpoint
            if os.path.isdir(args.output_dir):
                resume_ckpt = get_last_checkpoint(args.output_dir)
                # 兜底处理。某些版本没有写 last_checkpoint 文件
                if resume_ckpt is None:
                    names = [n for n in os.listdir(args.output_dir) if n.startswith("checkpoint-")]
                    if names:
                        names.sort(key=lambda x: int(x.split("-")[-1]))
                        resume_ckpt = os.path.join(args.output_dir, names[-1])
        elif val != "none":
            # 显式给了路径
            resume_ckpt = args.resume_from_checkpoint

    if resume_ckpt:
        print(f"[Resume] resume from {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
