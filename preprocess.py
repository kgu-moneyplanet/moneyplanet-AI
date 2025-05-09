import json

planet_descriptions = {
    "수성": "수성형 소비 성향: 충동적인 소비를 줄이고, 데이터를 기반으로 신중한 소비를 추구. 제품 리뷰, 가격 변동, 장기적 가치를 고려하여 현명한 소비 습관 형성.",
    "금성": "금성형 소비 성향: 감각적인 만족을 추구하되, 과소비를 줄이고 균형 잡힌 소비 습관 형성. 패션, 뷰티, 예술 등 취향을 반영하되, 예산을 고려한 소비 패턴 유지.",
    "지구": "지구형 소비 성향: 불필요한 지출을 줄이고, 철저한 예산 관리와 저축을 실천. 실용적이고 내구성이 좋은 제품을 선택하며, 경제적인 안정성을 높이는 소비 패턴 정착.",
    "화성": "화성형 소비 성향: 단순한 물건 구매를 줄이고, 삶의 질을 높이는 경험 소비를 늘리기. 충동 구매를 줄이고, 여행·레저·문화 활동 등 의미 있는 소비 습관 형성.",
    "목성": "목성형 소비 성향: 단순한 브랜드 소비를 넘어서, 자신에게 장기적으로 도움이 되는 소비를 추구. 프리미엄 제품보다는 자기 계발, 건강, 교육 등에 투자.",
    "토성": "토성형 소비 성향: 필요한 곳에만 돈을 쓰고, 가성비 높은 소비 습관을 기르기. DIY, 중고 거래, 구독 서비스 등을 활용해 낭비 없는 소비 생활 정착.",
    "천왕성": "천왕성형 소비 성향: 새로운 기술과 트렌드를 현명하게 활용하는 소비 습관 기르기. 최신 IT 기기, 구독 경제, 공유 서비스 등을 적극적으로 활용하여 효율적인 소비 패턴 형성.",
    "해왕성": "해왕성형 소비 성향: 소비를 통해 자신의 철학과 가치관을 반영하는 습관 기르기. 윤리적 소비, 친환경 브랜드, 사회적 가치를 중시하는 소비 습관을 정착."
}

# 1. 전처리 함수 정의
def preprocessing(example):
    user_input = example["input"]
    output = example["output"]

    planet = user_input["user_profile"]["planet"]
    instruction = (
        f"당신은 소비 성향 분석 AI입니다."
        f"다음 지출 내역을 분석하고 다음의 형식으로 답변하세요: 1.판단 결과, 2.판단 이유, 3.피드백.\n\n"
        f"소비 성향: {planet_descriptions.get(planet, '설명 없음')}"
    )

    full_prompt = (
        f"{instruction}\n\n"
        f"지출 내역:\n날짜: {user_input['spending_details']['date']}\n"
        f"금액: {user_input['spending_details']['amount']}원\n"
        f"카테고리: {user_input['spending_details']['category']}\n"
        f"설명: {user_input['spending_details']['description']}\n"
        f"소비 성향: {user_input['user_profile']['planet']}\n"
    )

    target = (
        f"1. 판단 결과: {output['classification_result']}\n"
        f"2. 판단 이유: {output['classification_reason']}\n"
        f"3. 피드백 내용: {output['feedback_content']}"
    )

    return {"prompt": full_prompt, "response": target}

# 2. 원시 데이터 로드 및 전처리
with open("dataset/dataset_example.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

prepro_data = [preprocessing(item) for item in raw_data]

# 3. 전처리 결과 저장
with open("dataset/prepro_dataset.json", "w", encoding="utf-8") as f:
    json.dump(prepro_data, f, ensure_ascii=False, indent=2)