import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re
from preprocess import planet_descriptions


# 모델 호출
def load_model():
    base_model_name = "davidkim205/komt-mistral-7b-v1"
    adapter_path = "./finetuned_model"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer

# 결과 파싱
def extract_fields(text: str):
    judgment = re.search(r"1\. 판단 결과:\s*(.*?)\n", text)
    reason = re.search(r"2\. 판단 이유:\s*(.*?)\n", text)
    feedback = re.search(r"3\. 피드백 내용:\s*(.*?)($|\n)", text)

    return {
        "judgment": judgment.group(1).strip() if judgment else "없음",
        "reason": reason.group(1).strip() if reason else "없음",
        "feedback": feedback.group(1).strip() if feedback else "없음"
    }

# 최종 예측
def predict_from_input(model, tokenizer, input_json: dict):
    user = input_json["user_profile"]
    spending = input_json["spending_details"]
    planet = user["planet"]
    desc = planet_descriptions.get(planet, "설명 없음")

    # 프롬프트 구성
    prompt = (
        f"당신은 소비 성향 분석 AI입니다. 다음 지출 내역을 분석하고 다음의 형식으로 답변하세요: "
        f"1. 판단 결과, 2. 판단 이유, 3. 피드백.\n\n"
        f"소비 성향: {desc}\n\n"
        f"지출 내역:\n"
        f"날짜: {spending['date']}\n"
        f"금액: {spending['amount']}원\n"
        f"카테고리: {spending['category']}\n"
        f"설명: {spending['description']}\n"
        f"소비 성향: {planet}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output[len(prompt):].strip()

    return extract_fields(generated)