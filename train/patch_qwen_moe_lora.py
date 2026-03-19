# patch_qwen_moe_lora.py
import torch.nn as nn
from moe_lora import MoELoRALinear
import re
TARGET_LINEAR_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Explicitly shield the visual side
BLOCKLIST_SUBSTRS = [
    "visual", "vision", "vit", "image", "mm_projector", "vision_tower", "clip", "merger"
]

# Allowed LLM path prefixes or keywords
ALLOWLIST_SUBSTRS = [
    "language_model",  
    "text_model",
    "decoder",
    ".model",          
]

def _is_target_linear(child_name: str) -> bool:
    return any(k in child_name for k in TARGET_LINEAR_KEYS)

def _is_llm_module_path(path: str) -> bool:
    p = path.lower()
    if any(b in p for b in BLOCKLIST_SUBSTRS):
        return False
    if p.startswith("model"):
        return True
    if any(a in p for a in ALLOWLIST_SUBSTRS):
        return True
    return False

def patch_model_with_moe_lora(
    model,
    num_experts=16,
    top_k=2,
    r=8,
    alpha=16,
    dropout=0.05,
    router_aux_weight=0.01,
    name_regex_list=None,          
    exclude_regex_list=None,       
    dryrun=False,                  
):

    replaced = 0
    hit_log = []

    include_patts = [re.compile(p) for p in (name_regex_list or [])]
    exclude_patts = [re.compile(p) for p in (exclude_regex_list or [])]


    def _safe_is_llm_path(path):
        try:
            return _is_llm_module_path(path)  
        except NameError:
            return True

    def _safe_is_target_linear(child_name):
        try:
            return _is_target_linear(child_name)  
        except NameError:
            # 默认命中常见投影层名
            return any(k in child_name for k in [
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj",
                "W_pack","w1","w2","w3"
            ])

    def _match_by_regex(fullname):
        inc_ok = True if not include_patts else any(p.search(fullname) for p in include_patts)
        exc_ok = not any(p.search(fullname) for p in exclude_patts)
        return inc_ok and exc_ok

    # Avoid duplicating the packaging of existing LoRA or MoE-LoRA
    def _skip_name(fullname):
        if "lora_" in fullname or fullname.endswith(".base_layer"):
            return True
        return False

    for module_path, module in list(model.named_modules()):
        if not _safe_is_llm_path(module_path):
            continue

        for child_name, child in list(module.named_children()):
            full_name = f"{module_path}.{child_name}" if module_path else child_name

            if _skip_name(full_name):
                continue
            if not isinstance(child, nn.Linear):
                continue
            if not _safe_is_target_linear(child_name):
                continue
            if not _match_by_regex(full_name):
                continue

            if len(hit_log) < 4000:
                hit_log.append(full_name)

            if dryrun:
                continue

            hidden = child.in_features
            new_layer = MoELoRALinear(
                base_linear=child,
                hidden_size=hidden,
                num_experts=num_experts,
                top_k=top_k,
                r=r,
                alpha=alpha,
                dropout=dropout,
                router_aux_weight=router_aux_weight,
            )

            setattr(module, child_name, new_layer)
            replaced += 1

    if replaced == 0 and not dryrun:
        print("[MoE-LoRA] WARNING patched 0 layers, check filters and layer names")
    else:
        if dryrun:
            print(f"[MoE-LoRA][dryrun] would patch {len(hit_log)} layers")
        else:
            more = "" if len(hit_log) <= 40 else f"\n  ... and {replaced - 40} more"
            head = "\n  - " + "\n  - ".join(hit_log[:40]) if hit_log else ""
            print(f"[MoE-LoRA] Patched {replaced} Linear layers in LLM{head}{more}")

    return model, hit_log

if __name__ == "__main__":
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoConfig,
        AutoProcessor,
        AutoModelForVision2Seq,
        Trainer,
        TrainingArguments,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "/home/ma-user/work/model/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(model)
    model, hit_log = patch_model_with_moe_lora(model, dryrun=False)
    print(len(hit_log), "layers would be patched")
    print("-------------------------------------------------------------------------------------")
    print(model)

