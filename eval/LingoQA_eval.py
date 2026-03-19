import os
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import re

CLIENT = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
TIMEOUT_S = 60
MODEL_NAME = "qwen3-max"

def infer_one(prompt):
    err: Optional[str] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are compassionate but fair and strict AI referee to judge True or False of student's answer."},
                    {"role": "user", "content": prompt},
                ],
                timeout=TIMEOUT_S,
            )
            text = res.choices[0].message.content if res and getattr(res, "choices", None) else ""
            try:
                m = re.search(r"\b(true|false)\b",text , flags=re.IGNORECASE)
                if not m:
                    raise ValueError("Neither True nor False found")

                return 1 if m.group(1).lower() == "true" else 0

            except (ValueError, TypeError):
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF ** (attempt - 1))
                    continue
                else:
                    return ""

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** (attempt - 1))
                continue
            else:
                return err

def batch_infer(tasks, concurrency: int = 8):
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(infer_one, s) for s in tasks]

        for fut in tqdm(futures, total=len(futures), desc="Batch inference"):
            results.append(fut.result())

    return results

get_gpt_score_prompt_template = """
Suppose you are driving, the student gives an answer of a question related to the driving scene.
Please judege the student's answer according to your deeply understanding in "Reference answer".
Question:{question}
Reference answer: {ref_answer}
Student's answer: {stu_answer}
"""
def build_prompt(question:str, ref_answer: str, stu_answer: str) -> str:
    return get_gpt_score_prompt_template.format(
        question = question,
        ref_answer=ref_answer,
        stu_answer=stu_answer
    )

GT_file_path = "/home/ma-user/work/dataset/FidelityAD/test_trafficQA/trafficQA.jsonl"
def LingoQA_get_GPT_res(item_list):
    ref_item_list = []
    with open(GT_file_path,"r",encoding="utf-8") as f:
        for la in f:              
            la = la.strip()
            if not la:
                continue
            item = json.loads(la)
            ref_item_list.append(item)      
    gpt_results = []
    concurrency = 128
    samples =[]
    for item, ref_item in zip(item_list,ref_item_list):
        question = ref_item["question"]
        ref_answer = ref_item["answer"]
        stu_answer = item["model_output"]
        get_gpt_score_prompt = build_prompt(question = question, ref_answer = ref_answer,stu_answer = stu_answer)
        samples.append(get_gpt_score_prompt)
    results = batch_infer(samples, concurrency=concurrency)
    # final_output = []
    # for item, score, GT in zip(item_list, results, ref_item_list):
    #     item["GT"] = GT
    #     item["score"] = score

    return results


if __name__ == '__main__':
    item_list = []
    ref_item_list = []
    item = {"task": "", "model_output": "No, a temposrary traffic light. It is showing red."}
    item_list.append(item)
    gpt_res = LingoQA_get_GPT_res(item_list)
    print(gpt_res)

