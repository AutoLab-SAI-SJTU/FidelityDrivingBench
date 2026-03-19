import re
import argparse
import json
import numpy as np
# import torch.nn as nn
# import language_evaluation
from multiprocessing import Pool
import sys
sys.path.append(".")
# from gpt_eval import GPTEvaluation

import argparse
import json
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from openai import OpenAI
from pathlib import Path
from collections import Counter

CLIENT = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
TIMEOUT_S = 60
MODEL_NAME = "qwen3-max"

# MAX_RETRIES = 3
# RETRY_BACKOFF = 2.0
# TIMEOUT_S = 60
# MODEL_NAME = "openai/gpt-5"


class evaluation_suit():
    def __init__(self,model_output, description):
        self.language_eval = language_evaluation.CocoEvaluator()
        self.model_output = model_output
        self.description = description

    def eval_language(self):
        results = self.language_eval.run_evaluation(self.model_output, self.description)
        return results


def run_task(model_output, GT):
    description = GT["description"]
    Evaluator = evaluation_suit(model_output,description)
    results = Evaluator.eval_language()

    return results

PROMPT_TEMPLATE = """Task: Tell me whether the following object is present or implied in the described scene.

Object to check: {object}

Scene description:
{scene_description}

Instruction:

Does the object "{object}" appear explicitly or is it implicitly implied based on the description?

Answer with one of the following:

Yes, explicitly present

Yes, implicitly implied

No, not present or implied

"""

def build_prompt(object: str, scene_description: str) -> str:
    return PROMPT_TEMPLATE.format(
        object=object,
        scene_description=scene_description,
    )


RETRY_BACKOFF = 2  # 退避基数：1s、2s、4s…

_ALLOWED = [
    "Yes, explicitly present",
    "Yes, implicitly implied",
    "No, not present or implied",
]
_ALLOWED_LOWER = [s.lower() for s in _ALLOWED]

def _is_valid_answer(text: str) -> bool:
    """判断text中是否包含任意一个允许答案（忽略大小写）。"""
    if not text:
        return False
    tl = text.lower()
    return any(opt in tl for opt in _ALLOWED_LOWER)

def _extract_label(text: str) -> Optional[str]:
    """从文本中提取第一个匹配到的标准标签（保持原样返回）。"""
    if not text:
        return None
    for pat in _ALLOWED:
        if pat.lower() in text.lower():
            return pat
    return None

def infer_one(prompt_path: list) -> Dict[str, Any]:
    MAX_RETRIES = 5
    err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a driving perception or scene understanding model."},
                    {"role": "user", "content": prompt_path[0]},
                ],
                timeout=TIMEOUT_S,
            )
            text = res.choices[0].message.content if res and getattr(res, "choices", None) else ""

            # 校验：若不包含三种标准答案之一，则视为无效并触发重试
            if not _is_valid_answer(text):
                raise ValueError("answer_not_in_expected_options")

            # 可选：提取并回填标准标签，便于下游稳定使用
            label = _extract_label(text)

            return {
                "img_path":prompt_path[1],
                "prompt": prompt_path[0],
                "eval_res": text,
                "label": label,   # 如需要只看结论，可读取这个字段
            }

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** (attempt - 1))
                continue
            else:
                return {
                    "img_path":prompt_path[1],
                    "prompt": prompt_path[0],
                    "eval_res": [],
                    "error": err
                }

def batch_infer(tasks, concurrency: int = 8) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(infer_one, s) for s in tasks]
        # 按提交的顺序收集结果
        for fut in tqdm(futures, total=len(futures), desc="Batch inference"):
            results.append(fut.result())

    return results

def load_samples(item_list:List,ref_item_list:List) -> List[Dict[str, Any]]:
    prompt_list = []
    for item, ref_item in zip(item_list,ref_item_list):
        scene_description = item["model_output"]
        obj_list = ref_item["noteworthy_objects"]
        for obj in obj_list:
            prompt = build_prompt(obj, scene_description)
            first_key = next(iter(item))
            prompt_list.append([prompt,item[first_key]])
    return prompt_list


VALID_LABELS = [
    "Yes, explicitly present",
    "Yes, implicitly implied",
    "No, not present or implied"
]

def count_labels(results):
    label_counter = Counter()
    for res in results:
        label = res["label"].strip()
        if label in VALID_LABELS:
            label_counter[label] += 1
        else:
            label_counter["Unknown/Invalid"] += 1  # 用于记录异常标签
    return label_counter


def run_GPT_eval(item_list:List,ref_item_list:List):
    concurrency = 16
    samples = load_samples(item_list,ref_item_list)
    total = len(samples)
    processed = 0
    results = batch_infer(samples, concurrency=concurrency)
    eval_res = count_labels(results)
    # print(eval_res)
    return results, eval_res


if __name__ == '__main__':
    from pprint import PrettyPrinter
    pprint = PrettyPrinter().pprint
    # model_output = "The image depicts an outdoor road scene with several notable elements:\n\n1. **Road**: The main feature is a two-lane paved road marked by white and orange lines, indicating lanes of travel.\n2. **Truck**: A large truck or trailer can be seen on the left side of the road heading away from the viewer's perspective (towards the horizon).\n3. **Sidewalk/Shoulder Area**: On the right-hand side near the edge of the pavement appears to be some sort of sidewalk area bordered by grassy patches.\n4. **Trees/Poles**: There’s dense greenery along both sides of the road consisting mainly of trees that line up closely together forming almost like a tunnel effect due to their proximity one another especially closer towards where they meet at ground level which could suggest utility poles as well given how tall these structures appear relative compared against other features within this frame such as buildings etc...\n5. **Buildings/Farm Structures**:\n   - In the background toward center-right there seems to exist what looks like agricultural storage silos alongside possibly part of farm infrastructure including barns/sheds likely used for livestock farming purposes judging off its size & shape.\n\n6. **Sky**: Overcast skies dominate much of upper portion giving overall gloomy atmosphere typical during early morning hours before sunrise/sunset depending upon exact time captured here but also suggesting possibility rain clouds moving through region shortly after photo was taken.\n\n\nIn summary: This photograph captures rural roadway environment featuring vehicles traveling down it while surrounded primarily natural landscape interspersed manmade constructions related agriculture sector visible further ahead into distance."
    model_output = "The scene shows a wet two-lane road curving slightly left under overcast conditions. A large gray truck is directly ahead in the same lane, with another vehicle visible further in the distance. The road has a solid yellow center line and white edge markings. On the left, there are utility poles and dense trees; on the right, an industrial or agricultural facility with silos and covered structures is visible. The wet pavement suggests recent rain, requiring cautious driving."
    GT = "{\n  \"description\": \"The scene shows a wet two-lane road curving slightly left under overcast conditions. A large gray truck is directly ahead in the same lane, with another vehicle visible further in the distance. The road has a solid yellow center line and white edge markings. On the left, there are utility poles and dense trees; on the right, an industrial or agricultural facility with silos and covered structures is visible. The wet pavement suggests recent rain, requiring cautious driving.\",\n  \"noteworthy_objects\": [\"Large truck ahead\", \"Wet road surface\", \"Solid yellow center line\"],\n  \"meta_actions\": [\"Follow lead vehicle\", \"Keep lane\"]\n}"
    results = run_task(model_output,GT)
    pprint(results)