import re

# 정제 코드
def clean_and_restore_spacing(text):
    """
    Qwen2.5-VL의 출력에서 시스템 메시지를 제거하고 띄어쓰기를 복원하는 함수.
    """
    # ✅ 1. "이 그림은" 또는 "이 장면은"이 나오기 전까지 모든 텍스트 제거
    text = re.sub(r".*?(이 그림은|이 장면은)", r"\1", text, flags=re.IGNORECASE | re.DOTALL)

    # ✅ 2. "이 이미지를 보고 ~ 설명하세요" 같은 프롬프트 제거
    prompt_text = "이 이미지를 보고 장면, 색채, 구도, 분위기, 주요 특징을 설명하세요."
    text = text.replace(prompt_text, "").strip()

    # ✅ 3. 연속된 공백을 한 개의 공백으로 변경
    text = re.sub(r"\s+", " ", text).strip()

    # ✅ 4. 한글과 영어/숫자 사이에 공백 추가 (자연스러운 띄어쓰기 복원)
    text = re.sub(r"([가-힣])([a-zA-Z0-9])", r"\1 \2", text)  # 한글 + 영어/숫자
    text = re.sub(r"([a-zA-Z0-9])([가-힣])", r"\1 \2", text)  # 영어/숫자 + 한글

    return text