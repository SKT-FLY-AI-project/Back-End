########################### SETP 2 : LLM #####################################

import openai
import os
import requests
import json
from dotenv import load_dotenv
from groq import Groq
import torch
import numpy as np
from PIL import Image

from app.utils.opencv_utils import get_color_name
from langchain.prompts import PromptTemplate

from app.models.llm_model import llm_model  # 미리 로드된 모델 불러오기
from app.services.text_processing import clean_and_restore_spacing

# Hugging Face 모델 캐시 경로 설정
os.environ['HF_HOME'] = "D:/huggingface_models"
client = Groq(api_key="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")

def generate_vlm_description_qwen(image_path):
    """미리 로드된 LLM 모델을 사용해 그림 설명 생성"""
    model, processor = llm_model.get_model()  # 로드된 모델 가져오기
    image = Image.open(image_path).convert("RGB").resize((512, 512))

    messages = [
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": "이 그림을 설명하세요."}]}
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=image, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    return clean_and_restore_spacing(processor.batch_decode(outputs, skip_special_tokens=True)[0])

########################### STEP 3 : 텍스트 생성 및 음성 변환 ###############################
from gtts import gTTS

def generate_rich_description(title, vlm_desc, dominant_colors, edges):
    """
    AI가 생성한 기본 설명을 기반으로 보다 풍부한 그림 설명을 생성하는 함수.
    """
    color_names = [get_color_name(c) for c in dominant_colors[:5]]
    dominant_colors_text = ", ".join(color_names)
    edges_detected = "명확히 탐지됨" if np.sum(edges) > 10000 else "불명확하게 탐지됨"

    prompt_template = PromptTemplate(
        input_variables=["title", "vlm_desc", "dominant_colors", "edges_detected"],
        template=f"""
        당신은 그림 설명 전문가입니다.  
        다음 그림에 대해 상세한 설명을 생성해주세요.
        시각장애인에게 설명할 수 있도록 자세하게 작성해 주세요.
        **단, 200자 ~ 500자 사이의 길이로만 생성해야 합니다!**

        - **제목:** "{title}"  
        - **VLM 기반 기본 설명:** "{vlm_desc}"   

        위 정보를 바탕으로 그림에 대한 상세한 설명을 작성해 주세요.  
        작품의 분위기, 색채, 구도, 표현 기법 등을 분석하고,  
        가능하다면 역사적, 예술적 배경도 함께 제공해 주세요.  
        설명은 반드시 **한글(가-힣)만 사용하여 작성해야 합니다.**  
        영어, 숫자, 특수문자는 포함할 수 없습니다.  
        """
    )

    formatted_prompt = prompt_template.format(
        title=title,
        vlm_desc=vlm_desc,
        dominant_colors=dominant_colors_text,
        edges_detected=edges_detected
    )

    completion = client.chat.completions.create(
        model="qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.5,
        max_tokens=1024,
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


########################### STEP 4 : 질문 답변 모드를 진행하는 함수 ###############################        
def answer_user_question(image_title, vlm_description, dominant_colors, edges):
    """사용자의 질문을 받아 LLM을 통해 답변을 생성하는 함수"""
    while True:
        user_question = input("\n❓ 추가 질문 (종료하려면 'exit' 입력): ")
        if user_question.lower() == "exit":
            print("📢 질문 모드 종료.")
            break
        
        # LLM에 전달할 프롬프트 생성
        prompt = f"""
        사용자는 '{image_title}' 작품에 대해 질문하고 있습니다.
        작품 설명: {vlm_description}
        주요 색상: {dominant_colors}
        엣지 감지 결과: {edges}
        
        사용자의 질문: "{user_question}"
        
        위 정보를 기반으로 사용자의 질문에 대해 상세하고 유익한 답변을 제공하세요.
        """
        
        # LLM을 이용한 답변 생성
        answer = generate_rich_description(image_title, prompt, dominant_colors, edges)
        print("\n💬 AI의 답변:")
        print(answer)

        # 음성 변환
        text_to_speech(answer, output_file=f"answer_{image_title}.mp3")
        
########################### STEP 5 : 질문 답변 모드를 진행하는 함수 ###############################
def start_vts_conversation(image_title, vlm_description): # 아직 RAG는 진행하지 않았습니다.
    """VTS 방식의 감상 대화를 진행하는 함수"""
    print("\n🖼️ VTS 감상 모드 시작!")
    
    while True:
        # LLM에 전달할 프롬프트 생성
        prompt = f"""
        사용자가 '{image_title}' 작품을 감상하고 있습니다.
        작품 설명: {vlm_description}

        사용자가 더 깊이 감상할 수 있도록 VTS(Visual Thinking Strategies) 방식의 질문을 하나씩 제공하세요.
        이전 질문과 연관되도록 새로운 질문을 제시하고, 감상자가 생각을 확장할 수 있도록 유도하세요.
        """
        
        # LLM을 이용한 VTS 질문 생성
        vts_question = generate_rich_description(image_title, prompt, [], [])
        
        # 사용자 입력 받기
        user_response = input(f"\n🎨 {vts_question} (종료하려면 'exit' 입력): ")
        if user_response.lower() == "exit":
            print("📢 VTS 감상 모드 종료.")
            break
