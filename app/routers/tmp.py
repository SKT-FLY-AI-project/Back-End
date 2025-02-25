import openai
import os
import requests
import cv2
import json
from dotenv import load_dotenv
from groq import Groq
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# 수제 번역 함수
from .data import translate

#from app.models.one_imageDetection.opencv_utils import get_color_name
from one_imageDetection.opencv_utils import get_color_name
from langchain.prompts import PromptTemplate

import random
from sentence_transformers import SentenceTransformer, util

# Hugging Face 모델 캐시 경로 설정
os.environ['HF_HOME'] = "D:/huggingface_models"
client = Groq(api_key="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")

# 모델 정보 설정
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 모델 로드 (FP16으로 변경)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=dtype,  # ✅ GPU는 FP16 사용, CPU는 FP32 사용
    device_map=device, # 원래는 auto. CPU 쓸거면 cpu로 바꿔야함.
    max_memory={0: "10GiB", "cpu": "30GiB"}
)

processor = AutoProcessor.from_pretrained(model_name)

########################## SETP 1 : 대화 검증 함수 ##############################
# 미술 정보 RAG에서 질문을 가져오는 함수 (검증용) : art_RAG_questions.json

def load_art_database():
    """예술 DB를 불러오는 함수"""
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    file_path = os.path.join(current_dir, "data", "labels_with_image_paths.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 예술 DB를 찾을 수 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# 작품 정보 검색 함수
def search_artwork_by_title(title):
    """
    작품 제목을 기반으로 RAG 문서에서 artist, period, webpage 정보를 검색하는 함수.
    """
    art_database = load_art_database()
    
    for entry in art_database:
        if entry["title"].lower() == title.lower():  # 대소문자 구분 없이 비교
            return {
                "artist": entry["artist_display_name"],
                "period": entry["period"],
                "webpage": entry["webpage"]
            }
    
    # 해당 작품이 없을 경우 Untitled로 처리
    return None

########################### SETP 2 : VLM (Vision to LLM) #####################################

import re

def clean_and_restore_spacing(text, prompt):
    """
    VLM(Qwen2.5-VL) 출력에서 프롬프트와 겹치는 부분을 자동 감지하여 제거하는 함수.
    """
    # ✅ 1. 프롬프트 내용을 그대로 포함하는 부분 제거
    prompt = prompt.strip()  # 앞뒤 공백 제거
    text = text.strip()  # 앞뒤 공백 제거

    # ✅ 2. 프롬프트와 출력이 겹치는 경우 삭제
    if prompt in text:
        text = text.replace(prompt, "").strip()

    # ✅ 3. "system", "You are a helpful assistant." 같은 AI 시스템 메시지 제거
    text = re.sub(r"(system|You are a helpful assistant\.|user)", "", text, flags=re.IGNORECASE)

    # ✅ 4. "assistant" 같은 응답 태그 제거 (ex: "assistant 1. 주요 객체")
    text = re.sub(r"assistant\s*\d*\.*", "", text, flags=re.IGNORECASE)

    # ✅ 5. 공백 및 줄바꿈 정리
    text = re.sub(r"\s+", " ", text).strip()

    # ✅ 6. 불필요한 기호(-, *, •) 정리 (일관되게 "-" 사용)
    text = re.sub(r"[•*]", "-", text)

    # ✅ 7. 한글과 영어/숫자 사이 띄어쓰기 복원
    text = re.sub(r"([가-힣])([a-zA-Z0-9])", r"\1 \2", text)  # 한글 + 영어/숫자
    text = re.sub(r"([a-zA-Z0-9])([가-힣])", r"\1 \2", text)  # 영어/숫자 + 한글

    return text


# 이미지 설명 VLM
def generate_vlm_description_qwen(image): # input이 이미지로 알고 있어서 image로 바꿈.
    # ✅ 이미지 로드 및 리사이징 (512x512)
    # 만약 image_path가 numpy 배열이라면:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # 이미지 받을 때 처리.
    else:
        image = Image.open(image).convert("RGB") # 만약 경로가 들어오면 그때 처리.
    image = image.resize((512, 512)) # 일단은 크기 정규화 했는데 추후 수정 필요.
    
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


    # ✅ 메시지 형식으로 변환 (apply_chat_template 사용)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # ✅ Chat Template 적용 (Qwen2.5-VL에서는 필수)
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # 이거 없으면 그냥 안돌아갑니다 진짜 중요함

    # ✅ 모델 입력 변환
    inputs = processor(
        text=[text_input],  # ✅ 변환된 텍스트 입력
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # ✅ 모델 실행 (토큰 수 최적화)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256) # 128로 하니까 좀 짤리는듯;

    # ✅ 결과 디코딩 및 세로 출력 문제 해결
    description = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    description = clean_and_restore_spacing(description, prompt) # 프롬프트를 받아야 정제 가능.

    return description


########################### STEP 3 : (VLM결과를 바탕으로) LLM ###############################
from gtts import gTTS

def generate_rich_description(title, vlm_desc, dominant_colors, edges):
    """
    AI가 생성한 기본 설명을 기반으로 보다 풍부한 그림 설명을 생성하는 함수.
    CNN이 인식한 작품이면 RAG 데이터를 사용하고, 인식하지 못하면 시각적 요소만 사용.
    """

    # 🔹 1. RAG에서 작품 정보 검색
    artwork_info = search_artwork_by_title(title)

    # 🔹 2. CNN이 작품을 인식하지 못한 경우 (Untitled 처리)
    if not artwork_info:
        print("🎨 CNN이 작품을 인식하지 못했습니다. 시각적 정보만 활용합니다.")
        color_names = [get_color_name(c) for c in dominant_colors[:5]]
        dominant_colors_text = ", ".join(color_names)
        edges_detected = edges #= "명확히 탐지됨" if np.sum(edges) > 10000 else "불명확하게 탐지됨"

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

        formatted_prompt = prompt_template.format(
            vlm_desc=vlm_desc,
            dominant_colors=dominant_colors_text,
            edges_detected=edges_detected
        )

        completion = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.5,
            max_tokens=512,
            top_p=0.95
        )

        return completion.choices[0].message.content.strip()
    
    # ✅ Fix: 색상 중복 제거
    color_names = [get_color_name(c) for c in dominant_colors[:5]]
    unique_colors = list(dict.fromkeys([color.strip() for name in color_names for color in name.split(',')]))
    colors_text = ", ".join(unique_colors)

    # 🔹 3. 작품을 인식한 경우 (RAG 정보 활용)
    prompt_variables = {
        "title": title,
        "artist": "Unidentified Artist", # 일단 작자 미상.
        "vlm_desc": vlm_desc,
        "dominant_colors": colors_text,
        #"edges_detected": "명확히 탐지됨" if np.sum(edges) > 10000 else "불명확하게 탐지됨"
    }

    if artwork_info:
        artist = artwork_info.get("artist")
        
        title_translations, artist_translations = translate.create_translation_mappings(title, artist)
        
        if title_translations:
            prompt_variables["title"] = title_translations
        if artist_translations:
            prompt_variables["artist"] = artist_translations
            
    if artwork_info.get("period"):
        prompt_variables["correct_period"] = artwork_info["period"]
    if artwork_info.get("webpage"):
        prompt_variables["webpage"] = artwork_info["webpage"]

    # 🔹 4. PromptTemplate을 사용하여 동적 프롬프트 구성
    # ✅ Prompt Template을 사용하여 프롬프트 구성
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

    # `None`이 포함된 key 제거
    filtered_prompt_variables = {k: v for k, v in prompt_variables.items() if v is not None}

    formatted_prompt = prompt_template.format(**filtered_prompt_variables)

    completion = client.chat.completions.create(
        model="qwen-2.5-32b",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.5,
        max_tokens=512,
        top_p=0.95
    )

    return completion.choices[0].message.content.strip()

def text_to_speech(text, output_file="output.mp3"):
    """
    생성된 텍스트를 음성 파일로 변환하여 저장하는 함수.
    """
    try:
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()
        tts = gTTS(text=text, lang='ko')
        tts.save(output_file)
        print(f"음성 파일이 '{output_file}'로 저장되었습니다.")
        os.system(f"start {output_file}")  # Windows (macOS: open, Linux: xdg-open)
    except Exception as e:
        print(f"음성 변환 중 오류 발생: {e}")
        
########################### STEP 5 : 대화 모드를 진행하는 함수 ###############################
# 1. RAG
# 1-1. VTS 질문지 RAG에서 질문을 가져오는 함수 (예시) : VTS_RAG_questions.json

# ✅ 1. 문장 유사도 분석을 위한 모델 로드 (KoBERT 사용)
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def load_vts_questions():
    """VTS 질문 파일을 불러오는 함수"""
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    file_path = os.path.join(current_dir, "data", "VTS_RAG_questions.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ VTS 질문 파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# ✅ 사용자의 입력 유형 분석 (작품 정보 요구 vs 감상 표현)
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


# ✅ 2. 사용자의 질문에 대한 답변 생성 (LLM 활용)
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
        

# ✅ 3. 사용자의 응답을 분석하여 적절한 VTS 질문 추천
def recommend_vts_question(user_response, previous_questions):
    """사용자의 응답과 가장 관련이 깊은 VTS 질문을 추천하는 함수"""
    vts_questions = load_vts_questions()

    # 이미 사용한 질문 제외
    available_questions = [q for q in vts_questions if q["question"] not in previous_questions]

    # 문장 임베딩 생성
    user_embedding = embedding_model.encode(user_response, convert_to_tensor=True)
    question_embeddings = embedding_model.encode([q["question"] for q in available_questions], convert_to_tensor=True)

    # 유사도 계산
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_match_idx = similarities.argmax().item()

    return available_questions[best_match_idx]["question"]


# ✅ LLM을 활용한 VTS 반응 및 질문 생성
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
        max_tokens=256,
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

    return reaction, question


# ✅ VTS 감상 대화 흐름 (피드백 + 질문 조합)
def start_vts_conversation(title, rich_description, dominant_colors, edges):
    """VTS 기반 감상 대화 진행 함수"""
    print("\n🖼️ VTS 감상 모드 시작!")

    conversation_history = []  # 대화 히스토리 저장
    user_response = input("🎨 작품을 보고 떠오른 느낌이나 궁금한 점을 말해주세요 (종료: exit): ")
    
    artwork_info = search_artwork_by_title(title)
    
    if artwork_info:
        artist = artwork_info.get("artist")
        
        title_translations, artist_translations = translate.create_translation_mappings(title, artist)
        title, artist = title_translations, artist_translations

    while user_response.lower() != "exit":
        
        # 질문 답변 종류 확인
        input_type = classify_user_input(user_response)
        
        if input_type == "info":
            # 🔹 대화 히스토리에 추가
            conversation_history.append(f"사용자: {user_response}")
            
            answer = answer_user_question(user_response, conversation_history, title, rich_description, dominant_colors)
            
            conversation_history.append(f"AI: {answer}")
            
            print("\n[정보 답변]")
            print(answer)
            
            user_response = input(f"🎨 혹시 더 궁금하신게 있으신가요? (종료: exit): ")
        
        elif input_type == "feeling":
            # 🔹 대화 히스토리에 추가
            conversation_history.append(f"사용자: {user_response}")

            # 🔹 AI 반응 및 질문 생성
            reaction, next_question = generate_vts_response(user_response, conversation_history)

            # 🔹 대화 히스토리에 추가
            conversation_history.append(f"AI: {reaction}")
            conversation_history.append(f"AI 질문: {next_question}")

            # 🔹 피드백 및 다음 질문 출력
            print(f"\n💬 {reaction}")
            user_response = input(f"🎨 {next_question} (종료: exit): ")

    print("📢 VTS 감상 모드 종료.")