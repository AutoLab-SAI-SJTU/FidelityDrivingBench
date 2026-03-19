"""
Inference for Qwen2.5-VL with MoE-LoRA and optional Mixture-of-Prompts (Prompt Bank + gating).

This script reuses utilities from train_qwen_moe_lora_v3.py when available.
It supports loading full checkpoints (safetensors shards) or PEFT adapters, and
automatically attaches PromptMixtureHead if such weights exist in the checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Transformers compat
try:
    from transformers import AutoModelForImageTextToText as HFImageTextModel
except Exception:
    from transformers import AutoModelForVision2Seq as HFImageTextModel
from transformers import AutoProcessor

# Import training utilities
from train_qwen_moe_lora_v4_flash import (
    TARGET_PATTERNS,
    _remap_and_filter_keys_for_moe as remap_and_filter_keys,
)
# Prefer PromptMixtureHead with top_k support from v4_flash; fallback to v3 version
# try:
from train_qwen_moe_lora_v4_flash import PromptMixtureHead as _PMH
# except Exception:
#     try:
#         from train_qwen_moe_lora_v3 import PromptMixtureHead as _PMH
#     except Exception:
#         _PMH = None
PromptMixtureHead = _PMH
from patch_qwen_moe_lora import patch_model_with_moe_lora
from moe_lora import MoELoRALinear


# Optional safetensors loader (reusing logic similar to v3 inference)
try:
    from safetensors.torch import load_file as _safe_load_file
    _HAS_SAFE = True
except Exception:
    _HAS_SAFE = False


def load_safetensors_state_dict(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    if not _HAS_SAFE:
        raise RuntimeError("需要安装 safetensors 以加载权重: pip install -U safetensors")

    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        shard_files = sorted({ckpt_dir / fname for fname in weight_map.values()})
        if not shard_files:
            raise FileNotFoundError(f"index 中没有列出分片文件: {index_path}")
        merged: Dict[str, torch.Tensor] = {}
        for sf in shard_files:
            merged.update(_safe_load_file(str(sf)))
        return merged

    safes = sorted(ckpt_dir.glob("model-*.safetensors"))
    if not safes and (ckpt_dir / "model.safetensors").exists():
        safes = [ckpt_dir / "model.safetensors"]
    if not safes:
        raise FileNotFoundError(f"未找到 safetensors 分片: {ckpt_dir}")
    merged: Dict[str, torch.Tensor] = {}
    for sf in safes:
        merged.update(_safe_load_file(str(sf)))
    return merged


def is_peft_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").exists()


def is_full_checkpoint_dir(p: Path) -> bool:
    patterns = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.safetensors",
    ]
    if any((p / name).exists() for name in patterns):
        return True
    if any(p.glob("pytorch_model-*.bin")):
        return True
    if any(p.glob("model-*.safetensors")):
        return True
    if any(p.glob("pytorch_model-*.safetensors")):
        return True
    return False


def _infer_prompt_head_shapes_from_sd(sd: Dict[str, torch.Tensor]) -> Tuple[Union[Tuple[int, int, int], None], str]:
    """
    Try to infer (K, P, D) and head prefix from state dict. Returns ((K,P,D), prefix) or (None, "").
    """
    prefixes = ["prompt_mixture_head.", "model.prompt_mixture_head."]
    for pref in prefixes:
        key = pref + "prompt_bank"
        if key in sd:
            w = sd[key]
            if isinstance(w, torch.Tensor) and w.dim() == 3:
                K, P, D = map(int, w.shape)
                return (K, P, D), pref
    return None, ""


def _attach_prompt_head_if_available(model: torch.nn.Module,
                                     sd_raw: Dict[str, torch.Tensor],
                                     args_prompt: Dict[str, Any]) -> None:
    """
    If checkpoint contains PromptMixtureHead weights, attach a head to model with matching shapes.
    Fallback: if user explicitly enables prompt mixture via CLI, attach with given K/P even if no weights found.
    """
    (shape, pref) = _infer_prompt_head_shapes_from_sd(sd_raw)
    if shape is not None:
        K, P, D = shape
        hid = int(getattr(getattr(model, "config", None), "hidden_size", None) or model.get_input_embeddings().weight.shape[1])
        if D != hid:
            # Hidden dim mismatch: still attach by trusting model hidden size, weights will raise later if incompatible
            pass
        # Try infer gate hidden from checkpoint
        gate0 = sd_raw.get(pref + "gate.0.weight", None)
        gate_hidden = int(gate0.shape[0]) if isinstance(gate0, torch.Tensor) and gate0.dim() == 2 else int(args_prompt.get("prompt_gate_hidden", 256))
        if PromptMixtureHead is None:
            print("[PromptMix] PromptMixtureHead not available; skip attaching head")
            model.prompt_mixture_head = None
            return
        try:
            model.prompt_mixture_head = PromptMixtureHead(
                hidden_size=hid,
                bank_size=K,
                prompt_len=P,
                gate_hidden=gate_hidden,
                top_k=int(args_prompt.get("prompt_top_k", 0) or 0),
            )
        except TypeError:
            # Fallback for older head without top_k
            model.prompt_mixture_head = PromptMixtureHead(
                hidden_size=hid,
                bank_size=K,
                prompt_len=P,
                gate_hidden=gate_hidden,
            )
        print(f"[PromptMix] head attached from ckpt shapes K={K} P={P} D={hid} gate_hidden={gate_hidden}")
        return

    # No weights found; attach only if explicitly requested
    if PromptMixtureHead is not None and args_prompt.get("prompt_mixture", False) and args_prompt.get("prompt_bank_size", 0) > 0 and args_prompt.get("prompt_len", 0) > 0:
        hid = int(getattr(getattr(model, "config", None), "hidden_size", None) or model.get_input_embeddings().weight.shape[1])
        try:
            model.prompt_mixture_head = PromptMixtureHead(
                hidden_size=hid,
                bank_size=int(args_prompt["prompt_bank_size"]),
                prompt_len=int(args_prompt["prompt_len"]),
                gate_hidden=int(args_prompt.get("prompt_gate_hidden", 256)),
                top_k=int(args_prompt.get("prompt_top_k", 0) or 0),
            )
        except TypeError:
            model.prompt_mixture_head = PromptMixtureHead(
                hidden_size=hid,
                bank_size=int(args_prompt["prompt_bank_size"]),
                prompt_len=int(args_prompt["prompt_len"]),
                gate_hidden=int(args_prompt.get("prompt_gate_hidden", 256)),
            )
        print(f"[PromptMix] head attached by CLI K={args_prompt['prompt_bank_size']} P={args_prompt['prompt_len']} D={hid} top_k={int(args_prompt.get('prompt_top_k', 0) or 0)}")
    else:
        model.prompt_mixture_head = None


def smart_load(base_model_path: str,
               lora_or_ckpt_path: Union[str, None],
               dtype: str,
               moe_args: Dict[str, Any],
               prompt_args: Dict[str, Any]):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() else torch.float16
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    try:
        tok = processor.tokenizer
        if tok is not None:
            tok.padding_side = "left"
    except Exception:
        pass
    model = HFImageTextModel.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    try:
        model.config.use_cache = True
    except Exception:
        pass

    if lora_or_ckpt_path:
        p = Path(lora_or_ckpt_path)
        if is_peft_adapter_dir(p):
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(p))
                print(f"[load] loaded PEFT adapter from {p}")
            except Exception as e:
                raise RuntimeError(f"加载 PEFT 适配器失败: {e}")
        elif is_full_checkpoint_dir(p):
            print(f"[load] detected full checkpoint at {p}  applying MoE-LoRA patch then loading weights")
            # patch MoE-LoRA first (same as training)
            model, _ = patch_model_with_moe_lora(
                model,
                num_experts=moe_args["num_experts"],
                top_k=moe_args["top_k"],
                r=moe_args["lora_r"],
                alpha=moe_args["lora_alpha"],
                dropout=moe_args["lora_dropout"],
                router_aux_weight=moe_args["router_aux_weight"],
                name_regex_list=TARGET_PATTERNS,
            )
            model.to(dtype=torch_dtype)
            # load raw weights to inspect shapes
            sd_raw = load_safetensors_state_dict(p)
            # attach prompt mixture head before filtering so that keys are kept
            _attach_prompt_head_if_available(model, sd_raw, prompt_args)
            # filter+remap keys then load
            sd = remap_and_filter_keys(sd_raw, model)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("  missing sample:", missing[:8])
            if unexpected:
                print("  unexpected sample:", unexpected[:8])
            model.to(dtype=torch_dtype)
        else:
            print(f"[load] path provided but neither PEFT adapter nor full checkpoint: {p}")

    model.eval()
    return model, processor

def load_images(image_root: Path, rel_paths: Union[str, Path, Sequence[Union[str, Path]]]):
    def _open_one(p: Union[str, Path]):
        try:
            rp = Path(p)
            path = rp if rp.is_absolute() else image_root / rp
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[warn] open image failed {p} {e}")
            return None

    if isinstance(rel_paths, (str, Path)):
        im = _open_one(rel_paths)
        return [im] if im is not None else []
    ims: List[Image.Image] = []
    for p in rel_paths:
        im = _open_one(p)
        if im is not None:
            ims.append(im)
    return ims


def build_mm_chat(prompt: str, n_img: int):
    import re
    content = []
    for _ in range(max(1, n_img)):
        content.append({"type": "image"})
    prompt = re.sub(r"^\s*<image>\s*", "", prompt, flags=re.IGNORECASE)
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _inject_prompt_mixture_for_generate(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """If model has prompt_mixture_head, compute fused prompt and insert at BOS; switch to inputs_embeds.
    Returns a new batch dict.
    """
    head = getattr(model, "prompt_mixture_head", None)
    if head is None:
        return batch

    input_ids = batch.get("input_ids", None)
    attn = batch.get("attention_mask", None)
    if input_ids is None:
        # nothing to do
        return batch
    # token embeddings
    embeds = model.get_input_embeddings()(input_ids)
    fused, _ = head(embeds, attn)
    B, T, D = embeds.shape
    P = fused.shape[1]
    pos = 1 if T >= 1 else 0
    left = embeds[:, :pos, :]
    right = embeds[:, pos:, :]
    new_embeds = torch.cat([left, fused, right], dim=1)

    # attention mask extension
    if attn is None:
        attn = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    left_m = attn[:, :pos]
    right_m = attn[:, pos:]
    ins_m = torch.ones((attn.shape[0], P), dtype=attn.dtype, device=attn.device)
    new_attn = torch.cat([left_m, ins_m, right_m], dim=1)

    out = dict(batch)
    out.pop("input_ids", None)
    out["inputs_embeds"] = new_embeds
    out["attention_mask"] = new_attn
    # labels 不用于推理
    out.pop("labels", None)
    return out


@torch.inference_mode()
def generate_one(model, processor, images, prompt, device,
                 max_new_tokens=128, do_sample=False, temperature=0.6, top_p=0.9,
                 num_beams=1, input_max_length=2048):
    conv = build_mm_chat(prompt, len(images))
    chat_text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    if images:
        batch = processor(images=images, text=chat_text, padding=False, truncation=False, return_tensors="pt")
    else:
        batch = processor(text=chat_text, padding=False, truncation=True, max_length=input_max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}

    # Record original input length before any embedding injection
    if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
        orig_in_len = int(batch["input_ids"].shape[-1])
    elif "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
        orig_in_len = int(batch["attention_mask"].shape[-1])
    else:
        orig_in_len = 0

    # Inject prompt mixture (if any)
    batch = _inject_prompt_mixture_for_generate(model, batch)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample, use_cache=True)
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    gen_ids = model.generate(**batch, **gen_kwargs)
    # Decode only newly generated tokens using original input length
    gen_seq = gen_ids[0]
    if gen_seq.shape[-1] > orig_in_len:
        new_tokens = gen_seq[orig_in_len:]
    else:
        new_tokens = gen_seq
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def detect_row_format(row: Dict[str, Any]) -> str:
    if "image" in row and "question" in row:
        return "trafficqa"
    if "task" in row:
        return "fidelityad"
    return "unknown"


def row_to_io(row: Dict[str, Any]) -> Tuple[List[str], Union[str, None], Dict[str, Any]]:
    fmt = detect_row_format(row)
    if fmt == "trafficqa":
        img = row.get("image", "")
        q = row.get("question", "")
        extra = {"answer": row.get("result", None)}
        return [img] if img else [], q, extra
    elif fmt == "fidelityad":
        imgs = row.get("task", [])
        return imgs, None, {"origin_path": row.get("origin_path", []), "noteworthy_objects": row.get("noteworthy_objects", [])}
    else:
        return [], json.dumps(row, ensure_ascii=False), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", type=str, required=True)
    ap.add_argument("--lora_path", type=str, default=None,
                    help="可以指向 PEFT 适配器目录 也可以指向 Trainer 的 checkpoint-XXXX 目录")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--output_path", type=str, default="predictions.jsonl")
    ap.add_argument("--prompt_str", type=str, default="<image>\nPlease describe the objects in the current scene.")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--input_max_length", type=int, default=4096)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    # MoE parameters for full checkpoint loading
    ap.add_argument("--num_experts", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--router_aux_weight", type=float, default=0.01)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Prompt mixture explicit attach (only used if checkpoint has no head but user forces it)
    ap.add_argument("--prompt_mixture", action="store_true")
    ap.add_argument("--prompt_bank_size", type=int, default=4)
    ap.add_argument("--prompt_len", type=int, default=8)
    ap.add_argument("--prompt_gate_hidden", type=int, default=512)
    ap.add_argument("--prompt_top_k", type=int, default=1)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moe_args = dict(
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_aux_weight=args.router_aux_weight,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    prompt_args = dict(
        prompt_mixture=bool(args.prompt_mixture),
        prompt_bank_size=int(args.prompt_bank_size),
        prompt_len=int(args.prompt_len),
        prompt_gate_hidden=int(args.prompt_gate_hidden),
        prompt_top_k=int(args.prompt_top_k),
    )

    model, processor = smart_load(args.base_model_path, args.lora_path, args.dtype, moe_args, prompt_args)
    model.to(device)

    data_path = Path(args.data_path)
    image_root = Path(args.image_root)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(data_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            images_rel, prompt_or_none, extra = row_to_io(row)
            images = load_images(image_root, images_rel) if images_rel else []
            the_prompt = prompt_or_none if prompt_or_none is not None else args.prompt_str

            ans = generate_one(
                model, processor, images, the_prompt, device,
                max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                temperature=args.temperature, top_p=args.top_p,
                num_beams=args.num_beams, input_max_length=args.input_max_length,
            )
            print(ans)

            fmt = detect_row_format(row)
            if fmt == "trafficqa":
                out = {
                    "image": row.get("image", ""),
                    "question": row.get("question", ""),
                    "model_output": ans,
                }
                if extra.get("answer", None) is not None:
                    out["answer"] = extra["answer"]
            elif fmt == "fidelityad":
                out = {
                    "origin_path": extra.get("origin_path", []),
                    "image_path": images_rel,
                    "noteworthy_objects": extra.get("noteworthy_objects", []),
                    "prompt": the_prompt,
                    "model_output": ans,
                }
            else:
                out = {"raw": row, "prompt": the_prompt, "prediction": ans}

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            n += 1
            if n % 50 == 0:
                print(f"[info] processed {n} samples")
    print(f"[done] wrote {n} lines to {out_path}")


if __name__ == "__main__":
    main()
