import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import openai

from app.app_config import get_settings
from app.modules.decision.interface.schema.decision_schema import AiInputData, OutputResponse

# -------------------- 모델 로딩 --------------------
BERT_MODEL_PATH = "app/models/bert_model"
XGB_MODEL_PATH = "app/models/xgb/xgb_model.pkl"
META_CLF_PATH = "app/models/xgb/meta_clf.pkl"
SCALER_PATH = "app/models/xgb/scaler.pkl"
OHE_PATH = "app/models/xgb/ohe.pkl"
LABEL_ENCODER_PATH = "app/models/xgb/label_encoder.pkl"

bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
xgb = joblib.load(XGB_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
ohe = joblib.load(OHE_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
meta_clf = joblib.load(META_CLF_PATH)

# -------------------- 행성 성향 사전 --------------------
planet_traits = {
    "수성": "충동적인 소비를 줄이고, 데이터를 기반으로 신중한 소비를 추구. 제품 리뷰, 가격 변동, 장기적 가치를 고려하여 현명한 소비 습관 형성.",
    "금성": "감각적인 만족을 추구하되, 과소비를 줄이고 균형 잡힌 소비 습관 형성. 패션, 뷰티, 예술 등 취향을 반영하되, 예산을 고려한 소비 패턴 유지.",
    "지구": "불필요한 지출을 줄이고, 철저한 예산 관리와 저축을 실천. 실용적이고 내구성이 좋은 제품을 선택하며, 경제적인 안정성을 높이는 소비 패턴 정착.",
    "화성": "단순한 물건 구매를 줄이고, 삶의 질을 높이는 경험 소비를 늘리기. 충동 구매를 줄이고, 여행·레저·문화 활동 등 의미 있는 소비 습관 형성.",
    "목성": "단순한 브랜드 소비를 넘어서, 자신에게 장기적으로 도움이 되는 소비를 추구. 프리미엄 제품보다는 자기 계발, 건강, 교육 등에 투자.",
    "토성": "필요한 곳에만 돈을 쓰고, 가성비 높은 소비 습관을 기르기. DIY, 중고 거래, 구독 서비스 등을 활용해 낭비 없는 소비 생활 정착.",
    "천왕성": "새로운 기술과 트렌드를 현명하게 활용하는 소비 습관 기르기. 최신 IT 기기, 구독 경제, 공유 서비스 등을 적극적으로 활용하여 효율적인 소비 패턴 형성.",
    "해왕성": "소비를 통해 자신의 철학과 가치관을 반영하는 습관 기르기. 윤리적 소비, 친환경 브랜드, 사회적 가치를 중시하는 소비 습관을 정착."
}

num_cols = ['amount', 'age', 'year', 'month', 'day']
cat_cols = ['category', 'planet', 'gender']

# -------------------- 메인 추론 함수 --------------------
def predict_from_input(ai_input_data: AiInputData) -> OutputResponse:
    input_data = {
        'description': ai_input_data.spending_details.description,
        'spending_reason': ai_input_data.spending_details.spending_reason,
        'job': ai_input_data.user_profile.job,
        'user_survey': ai_input_data.user_profile.user_survey,
        'amount': ai_input_data.spending_details.amount,
        'category': ai_input_data.spending_details.category,
        'planet': ai_input_data.user_profile.planet,
        'gender': ai_input_data.user_profile.gender,
        'age': ai_input_data.user_profile.age,
        'year': ai_input_data.spending_details.date.year,
        'month': ai_input_data.spending_details.date.month,
        'day': ai_input_data.spending_details.date.day,
    }

    input_data['planet_trait'] = planet_traits.get(input_data['planet'], "특징 없음")

    structured_feature = preprocess_structured(input_data, scaler, ohe, num_cols, cat_cols)
    final_text = make_final_text(input_data)

    # BERT 예측
    inputs = tokenizer(final_text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    bert_probs = torch.nn.functional.softmax(logits, dim=1).numpy()

    # XGBoost 예측
    xgb_probs = xgb.predict_proba(structured_feature)

    # Meta 분류기 앙상블
    stack_X = np.hstack([bert_probs, xgb_probs])
    pred_label_id = meta_clf.predict(stack_X)[0]
    pred_label = le.inverse_transform([pred_label_id])[0]

    # LLM 프롬프트 생성
    prompt = build_prompt(input_data)
    system_prompt = build_system_prompt(pred_label)

    openai.api_key = get_settings().OPENAPI_KEY
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.7,
        n=1,
    )
#    llm_output = response.choices[0].message.content
    llm_output = response.choices[0].message.content
    llm_output = " ".join(line.strip() for line in llm_output.strip().splitlines() if line)

    return OutputResponse(
        abc=convert_to_abc(pred_label),
        reason=pred_label,
        feedback=llm_output
    )

# -------------------- 유틸 함수 --------------------
def preprocess_structured(row, scaler, ohe, num_cols, cat_cols):
    num_features = scaler.transform([[row[col] for col in num_cols]])
    cat_features = ohe.transform([[row[col] for col in cat_cols]])
    return np.concatenate([num_features, cat_features], axis=1)

def make_final_text(row):
    return (
        f"날짜: {row['year']}년 {row['month']}월 {row['day']}일 [SEP] "
        f"설명: {row['description']} [SEP] "
        f"소비 사유: {row['spending_reason']} [SEP] "
        f"직업: {row['job']} [SEP] "
        f"사용자 설문: {row['user_survey']} [SEP] "
        f"행성 특성: {row['planet_trait']} [SEP] "
        f"금액: {row['amount']}원 [SEP] "
        f"카테고리: {row['category']} [SEP] "
        f"행성: {row['planet']} [SEP] "
        f"성별: {row['gender']} [SEP] "
        f"나이: {row['age']}세"
    )

def convert_to_abc(label: str) -> str:
    if label == "필요":
        return "A"
    elif label in ["선택", "선택적 소비"]:
        return "B"
    elif label == "불필요":
        return "C"
    return "F"

def build_prompt(row: dict) -> str:
    return f"""
    아래는 한 사용자의 소비 내역입니다.

    - 날짜: {row['year']}년 {row['month']}월 {row['day']}일
    - 카테고리: {row['category']}
    - 금액: {row['amount']}원
    - 소비 내역: {row['description']}
    - 소비 사유: {row['spending_reason'] or "해당 없음"}
    - 추구 행성: {row['planet']}
    - 행성 특성: {row['planet_trait']}
    - 성별: {row['gender']}
    - 나이: {row['age']}세
    - 직업: {row['job']}
    - 사용자 설문: {row['user_survey'] or "응답 없음"}
    """
# OpenAI API 호출 (ChatCompletion)
def build_system_prompt(pred_label: str) -> str:
    if pred_label == "필요":
        return """
        당신은 AI 가계부 앱의 조력자입니다. 사용자의 소비 성향은 8개의 행성 유형(예: 화성형, 금성형 등) 중 하나로 분류되며, 각 소비에 대해 LLM의 분류 결과에 따라 피드백을 제공합니다.

        지금 분석 중인 소비는 '필요'로 분류되었습니다.

        📌 당신의 역할은 다음과 같습니다:
        1. 소비가 왜 필요했는지 간결하게 설명합니다. (2~3문장)
           - 고려: 소비 금액, 이유, 사용자의 행성 성향, 직업/user_survey (관련 시)
           - 우주 여정 컨셉의 단어 사용: '자원 보충', '전략적 소비', '항해 중 에너지 확보' 등
        2. 절약 팁은 **절대 제공하지 마세요.**
        3. 마지막 문장은 컨셉에 맞춰 긍정적이고 창의적인 응원 문장으로 마무리합니다 (매번 다르게!).

        📋 출력 형식:
        [간결한 설명 문단]  
        [긍정적 응원 문장]
        """

    elif pred_label == "선택":
        return """
        당신은 AI 가계부 앱의 조력자입니다. 사용자의 소비 성향은 8개의 행성 유형 중 하나이며, 소비는 '선택'으로 분류되었습니다.

        📌 당신의 역할은 다음과 같습니다:
        1. 소비가 왜 '선택'으로 분류되었는지 설명합니다. (3~4문장)
           - 고려: 소비 금액, 이유, 사용자 성향, 직업/user_survey 정보 (필요 시)
           - '선택' 소비는 사용자의 가치관이나 상황에 따라 달라질 수 있습니다.
           - 우주 컨셉의 단어 활용 ('자원 선택', '항해 중 여유', '선택적 임무' 등)
        2. **절약 팁 1~2개**를 제시합니다.
           - 현실적인 대안
           - 자원 활용 측면의 조언
        3. 마지막에 컨셉에 맞춰 창의적이고 긍정적인 응원 문장을 작성하세요.

        📋 출력 형식: 
        [설명 문단]
        💡 절약 팁  
        - [대안1]  
        - [대안2]  
        [긍정적 응원 문장]
        """

    elif pred_label == "불필요":
        return """
        당신은 AI 가계부 앱의 조력자입니다. 사용자의 소비 성향은 8개의 행성 유형 중 하나이며, 이번 소비는 '불필요'로 분류되었습니다.

        📌 당신의 역할은 다음과 같습니다:
        1. 소비가 왜 '불필요'로 분류되었는지 설명합니다. (4~5문장)
           - 소비 금액, 이유, 사용자 행성 성향, user_survey 등을 고려합니다.
           - 특히 비합리적이거나 사치적인 부분을 지적하되, 비난이 아닌 조언의 톤을 유지합니다.
           - 우주 컨셉 단어 포함 ('자원 낭비', '항해 리스크', '전략적 분배 실패' 등)
        2. 절약 팁 1~2개 제시
           - 현명한 대체 소비나 생활 팁
           - 장기적 항해에 도움이 되는 방식
        3. 마지막은 컨셉에 맞춰 희망적이고 응원하는 문장으로 마무리합니다.

        📋 출력 형식:
        [설명 문단]
        💡 절약 팁  
        - [대안1]  
        - [대안2]  
        [긍정적 응원 문장]
        """

    else:
        raise ValueError(f"예측 라벨 '{pred_label}'은(는) 허용되지 않습니다.")
