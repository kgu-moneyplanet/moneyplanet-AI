import json
import re
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ========== 1. 데이터 로드 및 분할 ==========
print("데이터 로드 중...")
with open("dataset/prepro_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# ========== 2. 모델 및 토크나이저 로드 ==========
print("모델 다운로드 및 로딩 준비 중...")
model_name = "davidkim205/komt-mistral-7b-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 양자화 설정 (4bit + nf4 + float16)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4비트 양자화
    bnb_4bit_use_double_quant=True, # 더블 양자화 사용
    bnb_4bit_quant_type="nf4",  # nf4 양자화 방식
    bnb_4bit_compute_dtype=torch.float16    # 계산 시 16비트 사용
)

# 모델 로드 + 오프로드 설정 (로컬 환경에 맞춤)
print("모델 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",                     # 자동으로 GPU/CPU 분산
    trust_remote_code=True,
    #max_memory={0: "16GiB", "cpu": "32GiB"}  # GPU 16GB로 제한, CPU는 32GB 사용
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

print("모델 로딩 완료")

# ========== 3. PEFT (LoRA) 설정 ==========
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Mistral 구조에 맞게
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 모델로 변환
model = get_peft_model(model, peft_config)
print("LoRA 적용 완료")

# ========== 4. 토크나이즈 함수 ==========
def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(examples["response"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = targets["input_ids"]
    return inputs

print("데이터 토크나이징 중...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True) 

# ========== 5. 평가 지표 ==========
def extract_judgment(text):
    match = re.search(r"1\. 판단 결과:\s*(필요|선택|불필요)", text)
    return match.group(1) if match else "없음"  # 매칭 안될 경우 예외 처리

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 예측이 튜플 형태의(logits,)일 수 있으므로, 0번째 배열 사용
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # logits에서 예측된 토큰 ID 추출
    pred_ids = np.argmax(predictions, axis=-1)

    # 라벨의 -100 값을 pad_token_id로 변경 (디코딩 오류 방지)
    labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in label]
        for label in labels
    ]

    # 디코딩
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 판단 결과만 추출
    pred_classes = [extract_judgment(p) for p in decoded_preds]
    label_classes = [extract_judgment(l) for l in decoded_labels]

    return {
        "accuracy": accuracy_score(label_classes, pred_classes),
        "f1": f1_score(label_classes, pred_classes, average="weighted")
    }

# ========== 6. Trainer 설정 ==========
training_args = transformers.TrainingArguments(
    output_dir="./train_results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    max_steps=10,
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    eval_accumulation_steps=4,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ========== 7. 학습 시작 ==========
print("학습 시작!")
trainer.train()
print("학습 완료!")

# 학습 완료 후, 모델 저장
save_path = f"./finetuned_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("모델과 토크나이저가 저장되었습니다.")

# ========== 8. 평가 결과 출력 ==========
print("\n 최종 평가 결과:")
eval_results = trainer.evaluate()
print(f"평가 손실: {eval_results['eval_loss']:.3f}")
print(f"정확도: {eval_results['eval_accuracy']:.3f}")
print(f"F1 점수: {eval_results['eval_f1']:.3f}")
