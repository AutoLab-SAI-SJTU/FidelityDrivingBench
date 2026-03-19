import os
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# MAX_RETRIES = 5
# RETRY_BACKOFF = 2.0
# TIMEOUT_S = 60
# MODEL_NAME = "openai/gpt-5"

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
                    {"role": "system", "content": "You are compassionate but fair and strict AI judge which task is to grade student's answer. Please return a score only in figre format"},
                    {"role": "user", "content": prompt},
                ],
                timeout=TIMEOUT_S,
            )
            text = res.choices[0].message.content if res and getattr(res, "choices", None) else ""
            try:
                score = int(text.strip())
            except(ValueError, TypeError):
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF ** (attempt - 1))
                    continue
                else:
                    return (f"Cannot convert to int: {score!r}")
            return score
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
Suppose you are driving, the student gives a description of the driving scene.
Please evaluate the student's answer on a scale from 0 to 100 according to your deeply understanding in "Reference answer".
The higher score reflects precise alignment with the "Reference answer", well-supported reasoning, detailed description of the enviroment and accurate identification of road condition and objects. 
Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:
1. Accuracy and Alignment (50 points): Assign up to 20 points based on how the Student's answer correctly aligns with the description of the enviroment, road condition and objects refered in the Reference answer.
2. Conciseness and Clarity (30 points): Punish those inaccurate, ambiguous and irrelevant Student's answer that have nothing to do with the facts.
3. Bonus (20 points): If the Student's answer are correctly aligned in concise language, an additional bonus of 0-20 points will be given depending on the Clarity and Correctness level.
Reference answer: {ref_answer}
Student's answer: {stu_answer}
"""

def build_prompt(ref_answer: str, stu_answer: str) -> str:
    return get_gpt_score_prompt_template.format(
        ref_answer=ref_answer,
        stu_answer=stu_answer
    )

def get_GPT_score(item_list,ref_item_list):
    gpt_score_results = []
    concurrency = 8
    samples =[]
    for item, ref_item in zip(item_list,ref_item_list):
        ref_answer = ref_item["description"]
        stu_answer = item["model_output"]
        get_gpt_score_prompt = build_prompt(ref_answer = ref_answer,stu_answer = stu_answer)
        samples.append(get_gpt_score_prompt)
    results = batch_infer(samples, concurrency=concurrency)
    final_output = []
    for item, score, GT in zip(item_list, results, ref_item_list):
        item["GT"] = GT
        item["score"] = score

    return item_list


if __name__ == '__main__':
    item_list = []
    ref_item_list = []
    item = {"task": "/home/ma-user/work/dataset/FidelityAD/test/Val_regional_1288.png", "model_output": "Certainly! Here's a detailed description of the driving scene based on the image:\n\n---\n\n**Driving Scene Description:**\n\n**Time of Day:**\nThe image appears to have been captured during the evening or early night, as indicated by the dark blue sky and the illuminated streetlights.\n\n**Road and Environment:**\n- The road is a two-lane street lined with tall, leafless trees on both sides, suggesting it might be winter or late fall.\n- Streetlights are evenly spaced along the road, casting a warm, yellowish light that illuminates the path ahead.\n- The road is relatively quiet with minimal traffic. There is a car parked on the left side of the road, and a few other vehicles are visible further down the street.\n\n**Objects and People:**\n- On the right side of the road, there is a small kiosk or food cart with a green and yellow canopy. A person is standing next to the cart, possibly a vendor.\n- Near the kiosk, there are a few objects on the ground, including what appears to be a white stool or small table.\n- A pedestrian is walking on the sidewalk on the right side of the image, moving away from the camera.\n- There is a sign on the right side of the road, written in Chinese characters, indicating directions or information for drivers and pedestrians.\n\n**Safety Considerations:**\n- The road appears to be clear, with no immediate obstacles in the driving lane.\n- However, the parked car on the left side of the road and the objects near the kiosk could pose potential hazards if not noticed in time.\n- The pedestrian on the right side of the road should be kept in mind, especially if making any turns or lane changes.\n\n**Lighting and Visibility:**\n- The lighting conditions are decent, with streetlights providing sufficient illumination for safe driving.\n- The visibility of objects and people is clear, allowing for adequate reaction time.\n\n**Overall Impression:**\n- The scene is calm and quiet, typical of a suburban or less busy urban area during the evening.\n- Drivers should remain vigilant, especially around the kiosk area, to avoid any potential obstacles or pedestrians stepping into the road.\n\n---\n\nThis detailed description should provide a comprehensive understanding of the driving scene depicted in the image."}
    ref_result = {"description": "Nighttime urban road scene with streetlights illuminating the area. A lead vehicle is present ahead on the road, and there is a roadside vendor cart with people nearby on the right side. Traffic cones are visible on the road, indicating potential lane guidance or temporary setup.", "noteworthy_objects": ["concrete mixer truck", "stop sign", "construction barriers"], "meta_actions": ["Stop", "Follow lead vehicle"]}
    item_list.append(item)
    ref_item_list.append(ref_result)
    gpt_score_res = get_GPT_score(item_list,ref_item_list)
    print(gpt_score_res)

