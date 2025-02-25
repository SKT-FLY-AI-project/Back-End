
import os
import json
from PIL import Image
import numpy as np
import torch
from langchain.prompts import PromptTemplate

from app.utils.image_processing import get_color_name
from app.services.model_loader import llm_model  # 미리 로드된 모델 불러오기
from app.utils.text_processing import clean_and_restore_spacing
from app.config import client

async def generate_vlm_description_qwen(image_path):
    model, processor = llm_model.get_model()  # 로드된 모델 가져오기
    image = Image.open(image_path).convert("RGB").resize((512, 512))

    prompt = """
    이 그림을 보고 장면을 설명해주세요.  
    단순한 키워드가 아니라, 실제로 보고 이야기하듯이 짧은 문장으로 설명해주세요.  
    각 요소를 개별적으로 나열하는 것이 아니라, 전체적인 장면을 자연스럽게 묘사해주세요.  

    - 어떤 사물이 가장 눈에 띄나요?  
    - 사람들은 무엇을 하고 있나요?  
    - 색상과 빛의 흐름은 어떤 느낌을 주나요?  
    - 전체적인 분위기는 어떻게 표현될 수 있나요?  

    너무 길지 않게 2~3문장 정도로 설명해 주세요.
    설명은 반드시 **한글(가-힣)과 영어(a-z)만 사용하여 작성해야 합니다.**
    ⚠️ **한자(漢字)는 절대 포함하지 마세요.** ⚠️  
    한자가 포함될 경우, 다시 한글과 영어로만 설명해주세요.
    """
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=image, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    description = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    description = clean_and_restore_spacing(processor.batch_decode(outputs, skip_special_tokens=True)[0])
    
    return description

async def generate_rich_description(title, artist, correct_period, webpage, vlm_desc, dominant_colors, edges=[]):
    """
    AI가 생성한 기본 설명을 기반으로 보다 풍부한 그림 설명을 생성하는 함수.
    """
    color_names = [get_color_name(c) for c in dominant_colors[:5]]
    unique_colors = list(dict.fromkeys([color.strip() for name in color_names for color in name.split(',')]))
    colors_text = ", ".join(unique_colors)

    prompt_variables = {
        "title": title,
        "artist": artist,
        "correct_period": correct_period,
        "vlm_desc": vlm_desc,
        "dominant_colors": colors_text,
        "edges_detected": "명확히 탐지됨" if np.sum(edges) > 10000 else "불명확하게 탐지됨",
        "correct_period": correct_period,
        "webpage": webpage
    }
    
    prompt_template = ""
    if title is None:
        prompt_template = PromptTemplate(
            input_variables=["vlm_desc", "dominant_colors", "edges_detected"],
            template="""
            이 그림을 보면 어떤 느낌이 드나요?  
            색감과 분위기가 어떤 인상을 주는지 자연스럽게 설명해주세요.  

            - 그림을 보면 {vlm_desc} 같은 특징이 있어요.  
            - 색감은 {dominant_colors} 계열이 주를 이루고 있어요.  
            - 빛의 흐름을 보면 {edges_detected} 느낌이에요.  

            너무 딱딱한 설명보다는, 친구에게 그림을 소개하는 느낌으로  
            감성적이고 자연스럽게 이야기해 주세요.  
            200~300자 정도로 간결하면서도 풍부하게 표현해 주세요.
            설명은 반드시 **한글(가-힣)과 영어(a-z)만 사용하여 작성해야 합니다.**
            숫자, 특수문자, 한자는 포함할 수 없습니다.
            """
        )
    else:
        prompt_template = PromptTemplate(
            input_variables=list(prompt_variables.keys()),
            template="""
            {title}과(와) {artist}에 관한 정보만 검색하세요. 색상명({dominant_colors})은 정확히 주어진 그대로 사용해야 합니다.
            
            "{title}"라는 작품을 감상하고 있어요.  
            이 작품은 {artist}이(가) {correct_period} 시기에 제작한 작품이에요.  

            - 그림을 보면 {vlm_desc} 같은 특징이 있어요.  
            - 색감은 {dominant_colors} 계열이 주를 이루고 있어요. 색상 이름은 정확히 그대로 사용해주세요.
            
            이 작품의 분위기와 역사적 의미를 자연스럽게 설명해 주세요.  
            너무 학문적인 설명보다는, 편안한 대화처럼 표현해 주세요.
            200~300자 정도로 간결하고 감성적으로 작성해 주세요.
            설명은 반드시 **한글(가-힣)과 영어(a-z)만 사용하여 작성해야 합니다.**
            숫자, 특수문자, 한자는 포함할 수 없습니다.  

            {webpage}
            """
        )

    filtered_prompt_variables = {k: v for k, v in prompt_variables.items() if v is not None}

    formatted_prompt = prompt_template.format(**filtered_prompt_variables)

    completion = client.chat.completions.create(
        model="qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.5,
        max_tokens=1024,
        top_p=0.95
    )

    return completion.choices[0].message.content.strip()


def load_vts_questions():
    """VTS 질문 파일을 불러오는 함수"""
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    file_path = os.path.join(current_dir, "data", "VTS_RAG_questions.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ VTS 질문 파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def classify_user_input(user_input):
    """
    사용자의 입력이 작품 설명을 요구하는지(1-1) vs 자신의 감상을 말하는지(1-2) 분류하는 함수.
    """
    keywords_info = ["이 작품", "설명", "배경", "작가", "의미", "당시 상황"]
    keywords_feeling = ["느낌", "분위기", "인상적", "마음에 들어", "생각", "의견"]

    if any(keyword in user_input for keyword in keywords_info):
        return "info"  # 작품 설명 요청 (1-1)
    elif any(keyword in user_input for keyword in keywords_feeling):
        return "feeling"  # 감상 표현 (1-2)
    return "unknown"

def answer_user_question(user_response, conversation_history, title, artist, rich_description):
    # 🔹 대화 맥락 정리
    context = "\n".join(conversation_history[-3:])  # 최근 3개만 유지 (메모리 최적화)
    
    conversation_history.append(f"사용자: {user_response}")

    prompt = f"""
            사용자는 '{artist}'의 '{title}' 작품에 대해 질문하고 있습니다.
            이전 대화 : 
            {context}
            사용자의 질문 : 
            "{user_response}"
            
            작품 설명 : "{rich_description}"
            사용자의 질문: "{user_response}"
            
            위 정보를 기반으로 상세하고 유익한 답변을 제공하세요.
            """

    completion = client.chat.completions.create(
        model="qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=256,
        top_p=0.95
    )
    
    return completion.choices[0].message.content.strip()

def generate_vts_response(user_input, conversation_history):
    """
    사용자의 입력과 대화 히스토리를 기반으로 적절한 반응과 질문을 생성하는 함수.
    """
    # 🔹 대화 맥락 정리
    context = "\n".join(conversation_history[-3:])  # 최근 3개만 유지 (메모리 최적화)

    prompt = f"""
    사용자가 미술 작품을 감상하고 있습니다.
    이전 대화:
    {context}

    사용자의 입력:
    "{user_input}"

    AI의 역할:
    1. 사용자의 감상에 대해 적절한 반응을 제공합니다.
    2. 새로운 질문을 생성하여 자연스럽게 대화를 이어갑니다.

    AI의 응답 형식:
    1. 반응: (사용자의 감상을 반영한 피드백)
    2. 질문: (VTS 기반의 적절한 추가 질문)
    """

    completion = client.chat.completions.create(
        model="qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=150,
        top_p=0.95
    )

    response = completion.choices[0].message.content.strip()
    
    # 🔹 응답을 "반응 + 질문"으로 분리
    try:
        response_parts = response.split("\n")
        reaction = response_parts[0].strip() if response_parts else "흥미로운 생각이에요."
        question = response_parts[1].strip() if len(response_parts) > 1 else "이 작품을 보고 어떤 점이 가장 인상적이었나요?"
    except:
        reaction, question = response, "이 작품을 보고 어떤 점이 가장 인상적이었나요?"

    return reaction[7:], question[7:]