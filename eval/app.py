# app.py
import os
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from eval import run_GPT_eval   # Noteworthy objects eval
from get_gpt_score import get_GPT_score 
from LingoQA_eval import LingoQA_get_GPT_res # TrafficQA_eval

app = FastAPI(title="JSONL Processor Service", version="1.0.0")

SAVE_DIR = Path("./work/server_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class GPTResult(BaseModel):
    res:str

class GPT_Score(BaseModel):
    res:float

class GPT_acc(BaseModel):
    res:str


@app.post("/gpt_eval", response_model=GPTResult)
async def gpt_eval(
    file: UploadFile = File(..., description="上传一个 .jsonl 文件"),
    output_name: Optional[str] = Form(None, description="可选：结果txt文件名，如 result.txt"),
):
    base_name = output_name if output_name else f"{Path(file.filename).stem}_{uuid.uuid4().hex[:8]}.txt"
    out_path = SAVE_DIR / base_name
    obj_list = []
    ref_obj_list = []
    check_file_path = "/home/ma-user/work/dataset/FidelityAD/test_annotations_output.jsonl"

    with out_path.open("w", encoding="utf-8") as fout,\
        open(check_file_path, 'r', encoding='utf-8') as f_in:
        try:
            await file.seek(0)
            for raw_line, ref_line in zip(file.file,f_in):
                line = raw_line.decode("utf-8").strip()
                ref_line = ref_line.strip()
                ref_obj = json.loads(ref_line)
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    # 非法 json 行，写个错误信息进去，或选择跳过
                    fout.write(f"[BAD JSON LINE]: {e}\n")
                    continue
                obj_list.append(obj)
                ref_obj_list.append(ref_obj["result"])
            eval_res_list,label_counter = run_GPT_eval(obj_list,ref_obj_list)
            for eval_item in eval_res_list:
                fout.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
            return GPTResult(res = str(label_counter))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {e}")

@app.post("/gpt_score", response_model=GPT_Score)
async def gpt_score(
    file: UploadFile = File(..., description="上传一个 .jsonl 文件"),
    output_name: Optional[str] = Form(None, description="可选：结果txt文件名，如 result.txt"),
):
    base_name = output_name if output_name else f"{Path(file.filename).stem}_{uuid.uuid4().hex[:8]}.txt"
    out_path = SAVE_DIR / base_name
    obj_list = []
    ref_obj_list = []
    check_file_path = "/home/ma-user/work/dataset/FidelityAD/test_annotations_output.jsonl"

    lines_processed = 0
    total = 0
    with out_path.open("w", encoding="utf-8") as fout,\
        open(check_file_path, 'r', encoding='utf-8') as f_in:
        try:
            await file.seek(0)
            for raw_line, ref_line in zip(file.file,f_in):
                line = raw_line.decode("utf-8").strip()
                ref_line = ref_line.strip()
                ref_obj = json.loads(ref_line)
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    # 非法 json 行，写个错误信息进去，或选择跳过
                    fout.write(f"[BAD JSON LINE]: {e}\n")
                    continue
                obj_list.append(obj)
                ref_obj_list.append(ref_obj["result"])
                lines_processed += 1
            gpt_score_list = get_GPT_score(obj_list,ref_obj_list)
            
            for gpt_score in gpt_score_list:
                fout.write(json.dumps(gpt_score, ensure_ascii=False) + "\n")
                total += int(gpt_score["score"])
            avg_score = total/lines_processed
            return GPT_Score(res = f"{avg_score:.2f}")         
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {e}")

@app.post("/gpt_acc", response_model=GPT_acc)
async def gpt_acc(
    file: UploadFile = File(..., description="上传一个 .jsonl 文件"),
    output_name: Optional[str] = Form(None, description="可选：结果txt文件名，如 result.txt"),
):
    base_name = output_name if output_name else f"{Path(file.filename).stem}_{uuid.uuid4().hex[:8]}.txt"
    out_path = SAVE_DIR / base_name
    obj_list = []
    lines_processed = 0
    true_num = 0 
    with out_path.open("w", encoding="utf-8") as fout:
        try:
            await file.seek(0)
            for raw_line in file.file:
                line = raw_line.decode("utf-8").strip()
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    # 非法 json 行，写个错误信息进去，或选择跳过
                    fout.write(f"[BAD JSON LINE]: {e}\n")
                    continue
                obj_list.append(obj)
                lines_processed += 1
            gpt_acc_list = LingoQA_get_GPT_res(obj_list)
            for gpt_acc in gpt_acc_list:
                fout.write(json.dumps(gpt_acc, ensure_ascii=False) + "\n")
                if gpt_acc == 1:
                    true_num += 1
            
            return GPT_acc(res = f"Accuracy:{true_num/lines_processed}")         
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {e}")


# start server command: uvicorn app:app --host 0.0.0.0 --port 10086 --reload
# request command: 
"""
curl -F "file=@/home/ma-user/modelarts/user-job-dir/A100/FidelityAD/eval/test_input.jsonl" \
     -F "output_name=result.jsonl" \
     http://192.168.251.174:8000/process_jsonl

curl -F "file=@/home/ma-user/work/dataset/FidelityAD/recogdrive_output_test.jsonl" \
     -F "output_name=result.jsonl" \
     http://192.168.251.174:8000/process_jsonl

curl -F "file=@/home/ma-user/work/dataset/FidelityAD/qwenvl72b_output.jsonl" \
     -F "output_name=qwenvl72b_res.jsonl" \
     http://192.168.251.174:8000/process_jsonl

curl -F "file=@/home/ma-user/modelarts/user-job-dir/A100/FidelityAD/eval/test_input.jsonl" \
     -F "output_name=gpt_result.jsonl" \
     http://192.168.251.174:8000/gpt_eval

curl -F "file=@/home/ma-user/work/dataset/FidelityAD/recogdrive_output_test.jsonl" \
     -F "output_name=recogdrive_gpt_result.jsonl" \
     http://192.168.251.174:8000/gpt_eval

curl -F "file=@/home/ma-user/work/dataset/FidelityAD/qwenvl72b_output.jsonl" \
        -F "output_name=recogdrive_gpt_result.jsonl" \
        http://192.168.251.174:8000/gpt_eval

"""