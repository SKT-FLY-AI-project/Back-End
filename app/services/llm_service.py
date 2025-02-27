
import os
import json
from PIL import Image
import numpy as np
import torch
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
import asyncio
import pickle

from app.utils.image_processing import get_color_name
from app.services.model_loader import llm_model  # 미리 로드된 모델 불러오기
from app.utils.text_processing import clean_and_restore_spacing
from app.config import client, PICKLE_PATH

embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
import sys

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
    if image is None:
        raise ValueError("Image is None!")
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Text input: {text_input}")
    print(f"Image type: {type(image)}")

    inputs = processor(text=[text_input], images=image, return_tensors="pt", padding=True)
    print(f"Inputs before moving to device: {inputs}")


    print("🚀 Debugging inputs before moving to GPU:")
    sys.stdout.flush()  # 강제 출력

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"🔹 {key} shape: {value.shape}, dtype: {value.dtype}, device: {value.device}")
            sys.stdout.flush()  # 강제 출력

            print(f"   Min: {value.min()}, Max: {value.max()}")
            sys.stdout.flush()  # 강제 출력

            print(f"   Any NaN? {torch.any(torch.isnan(value))}")
            sys.stdout.flush()  # 강제 출력

            print(f"   Any Inf? {torch.any(torch.isinf(value))}")
            sys.stdout.flush()  # 강제 출력
            print(f"   Unique values: {torch.unique(value)[:10]}")  # 유니크 값 일부 확인
            sys.stdout.flush()  # 강제 출력
    
    # 🚀 2. `pixel_values` 정규화 (모델이 기대하는 범위로 변환)
    if "pixel_values" in inputs:
        print("🚨 Warning: Normalizing pixel_values")
        sys.stdout.flush()
        min_val, max_val = inputs["pixel_values"].min(), inputs["pixel_values"].max()
        
        # [-1, 1]로 정규화
        inputs["pixel_values"] = 2 * ((inputs["pixel_values"] - min_val) / (max_val - min_val)) - 1

    # 🚀 3. `int64` 값 유지 (잘못 변환되지 않도록)
    # 모델이 기대하는 dtype에 맞게 변환
    inputs["input_ids"] = inputs["input_ids"].long()  # 일반적으로 long 유지
    inputs["attention_mask"] = inputs["attention_mask"].long()
    inputs["image_grid_thw"] = inputs["image_grid_thw"].long()
    
    # 만약 `images` 값이 포함된다면 float32로 변환
    if "images" in inputs:
        inputs["images"] = inputs["images"].float()
    
    # 🚀 4. GPU로 이동 (`pixel_values`만 `float32` 변환)
    device = model.device
    inputs = {
        k: v.to(device, dtype=torch.float32) if k == "pixel_values" else v.to(device)
        for k, v in inputs.items() if isinstance(v, torch.Tensor)
    }
    
    print("✅ Successfully moved inputs to GPU.")
    sys.stdout.flush()


    loop = asyncio.get_event_loop()
    with torch.no_grad():
        outputs = await loop.run_in_executor(
            None, lambda: model.generate(**inputs, max_new_tokens=256)  # 512 → 256으로 변경
        )


    description = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    description = clean_and_restore_spacing(processor.batch_decode(outputs, skip_special_tokens=True)[0])
    
    return description

async def generate_rich_description(title, artist, correct_period, vlm_desc, dominant_colors, edges=[]):
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
        "correct_period": correct_period
    }
    
    prompt_template = ""
    if artist is None:
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
    
# ✅ 사용자의 입력 유형 분석 (작품 정보 요구 vs 감상 표현)
import re

def classify_user_input(user_input):
    """
    미술작품 감상 관련 대화를 정보 요청과 감상 표현으로 분류하는 함수
    """
    keywords_info = {
       # 작품 기본 정보
       "무엇", "뭐", "어떤", "어떻게", "언제", "어디서", "누가", "왜",
       "설명해", "알려줘", "가르쳐", "말해줘", "궁금해",
       
       # 작품 상세 정보
       "제목", "년도", "시기", "크기", "소장", "전시", "보관",
       "캔버스", "유화", "수채화", "판화", "소재", "재료",
       "채색", "물감", "안료", "염료", "붓", "도구",
       
       # 작가 관련
       "작가", "화가", "미술가", "예술가", "대가", "장인",
       "유파", "화파", "학파", "스승", "제자", "영향",
       "동시대", "같은 시기", "활동", "작품세계",
       
       # 작품 해석/의미
       "의미", "상징", "해석", "메시지", "주제", "내용",
       "모티프", "소재", "題材", "타이틀", "제재",
       "구성", "구도", "배치", "화면", "장면",
       
       # 미술사/맥락
       "미술사", "예술사", "당시", "시대", "운동", "사조",
       "유파", "화파", "르네상스", "바로크", "인상주의",
       "표현주의", "추상", "모더니즘", "컨템포러리",
       
       # 기법/표현
       "기법", "테크닉", "기교", "화법", "묘사", "표현",
       "원근법", "명암법", "농담", "채색", "스케치",
       "데생", "드로잉", "터치", "붓질", "획", "선", "면",
       "명암", "음영", "그라데이션", "질감", "마티에르"
   }
   
    keywords_feeling = {
        # 감상/인상
        "느낌", "감정", "인상", "감동", "정서", "감성",
        "분위기", "무드", "아우라", "기운", "기세",
        "여운", "울림", "전율", "충격", "감흥",
        
        # 시각적 반응
        "보이다", "눈에 띄다", "시선", "눈길", "주목",
        "색감", "색채", "컬러", "톤", "색조", "색상",
        "화려", "은은", "강렬", "부드럽", "차분", "고요",
        
        # 긍정적 평가
        "좋다", "멋지다", "아름답", "우아", "품격", "기품",
        "뛰어나다", "탁월", "완벽", "대단", "훌륭",
        "세련", "정교", "섬세", "우수", "탁월", "빼어나",
        
        # 주관적 해석
        "생각", "보입니다", "싶습니다", "것 같아요",
        "연상", "떠올라요", "기억", "추억", "경험",
        "공감", "이해", "와닿다", "닮았다", "유사",
        
        # 예술적 평가
        "조화", "균형", "통일", "대비", "리듬", "율동",
        "구성", "완성도", "독창", "창의", "혁신", "개성",
        "특색", "특징", "개성", "독특", "참신", "획기적",
        
        # 감정 표현
        "기쁘다", "슬프다", "평온", "불안", "고요", "역동",
        "즐겁다", "우울하다", "따뜻", "차갑", "밝다", "어둡다",
        "부드럽다", "거칠다", "강하다", "약하다"
    }

    def contains_keyword(text, keyword):
        """
        더 정확한 키워드 매칭을 위한 함수
        """
        pattern = fr'(^|[^\w]){keyword}([^\w]|$)'  # 'r' 접두사 추가
        return bool(re.search(pattern, text))

    def analyze_sentence_ending(text):
        """
        문장 끝맺음을 분석하여 의도 파악을 돕는 함수
        """
        if text.strip().endswith(('?', '까요?', '나요?', '죠?')):
            return "info"
        elif text.strip().endswith(('네요', '어요', '아요', '!', '~')):
            return "feeling"
        return None

    # 키워드 매칭 확인 (정확한 매칭 사용)
    info_count = sum(1 for keyword in keywords_info if contains_keyword(user_input, keyword))
    feeling_count = sum(1 for keyword in keywords_feeling if contains_keyword(user_input, keyword))
   
    # 문장 끝맺음 분석
    ending_type = analyze_sentence_ending(user_input)
    
    # 가중치를 둔 최종 분류
    if ending_type:
        # 문장 끝맺음이 명확한 경우, 이를 우선 고려
        if ending_type == "info" and info_count > 0:
            return "info"
        elif ending_type == "feeling" and feeling_count > 0:
            return "feeling"
    
    # 키워드 기반 분류
    if info_count > feeling_count:
        return "info"
    elif feeling_count > info_count:
        return "feeling"
    elif info_count > 0 and info_count == feeling_count:
        # 동점인 경우 문장 끝맺음으로 판단
        return ending_type if ending_type else "mixed"
    else:
        # 키워드가 없는 경우 문장 끝맺음으로 판단
        return ending_type if ending_type else "unknown"

# 미술작품 RAG 관련 함수
# 1. 텍스트 데이터 로드
# 미리 전처리된 데이터를 pickle 파일에서 불러오기
async def load_precomputed_data(pickle_file=None):
    """
    저장된 pickle 파일에서 전처리된 데이터를 불러와 반환합니다.
    만약 pickle_file 인자가 주어지지 않으면, 현재 스크립트 기준으로 './data/precomputed_data.pkl' 경로를 사용합니다.
    """
    # if pickle_file is None:
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     pickle_file = os.path.join(current_dir, "data", "precomputed_data.pkl")
        
    with open(PICKLE_PATH, 'rb') as f:
        precomputed_data = pickle.load(f)
    return precomputed_data

def find_top_k_similar(query_sentence, sentence_dict, embeddings, model, top_k=20, threshold=0.65):
    """
    query_sentence와 임베딩된 문장들 사이의 코사인 유사도를 계산하여,
    상위 top_k 개의 문장(인덱스, 문장, 유사도) 리스트를 반환합니다.
    임계치보다 낮은 유사도는 결과에서 제외합니다.
    """
    query_embedding = model.encode(query_sentence, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_k_scores, top_k_indices = torch.topk(cosine_scores, k=top_k)
    
    results = []
    for score, idx in zip(top_k_scores, top_k_indices):
        score = score.item()
        idx = idx.item()
        sentence_key = str(idx + 1)  # 인덱스는 1부터 시작한다고 가정
        if score >= threshold:
            results.append((sentence_key, sentence_dict[sentence_key], score))
    return results

# 2. 관련 문서 검색 함수
def retrieve_relevant_info(precomputed_data, query, title, artist, top_k=20, threshold=0.45):
    """
    미리 전처리된 데이터에서, title과 artist 정보를 포함한 검색 쿼리와 가장 유사한 상위 top_k 개의 문장을 찾아
    각 문장의 출처 인덱스와 함께 문자열 형태로 반환합니다.
    """
    # title과 artist 정보를 포함한 통합 쿼리 생성
    combined_query = f"{title} {artist} {query}"
    
    sentence_dict = precomputed_data["sentence_dict"]
    embeddings = precomputed_data["embeddings"]
    results = find_top_k_similar(combined_query, sentence_dict, embeddings, embedding_model, top_k=top_k, threshold=threshold)
    
    if results:
        formatted_results = "\n".join([f"Source [{key}]: {sentence} (유사도: {sim_score:.2f})"
                                       for key, sentence, sim_score in results])
        return formatted_results
    else:
        return "유사한 관련 미술 자료를 찾지 못했습니다."

def answer_user_question(user_response, conversation_history, title, artist, rich_description, precomputed_data):
    """RAG를 활용하여 미술 작품 관련 질문에 답변하는 함수"""
    # 대화 맥락 정리
    context = "\n".join(conversation_history[-3:])  # 최근 3개만 유지 (메모리 최적화)
    print(context)
    # RAG: 질문에 관련된 정보 검색
    retrieved_info = retrieve_relevant_info(precomputed_data, user_response, title, artist, top_k=20, threshold=0.45)
    print(retrieved_info)
    
    # 프롬프트 구성
    prompt = f"""
    사용자는 '{artist}'의 '{title}' 작품에 대해 질문하고 있습니다.
    
    이전 대화: 
    {context}
    
    사용자의 질문: 
    "{user_response}"
    
    작품 설명: 
    "{rich_description}"
    
    관련 미술 자료:
    {retrieved_info}
    
    위 정보를 기반으로 유익한 답변을 제공하세요.
    관련 미술 자료에서 찾은 정보를 활용하되, 작품과 직접 관련이 없는 내용은 제외하세요.
    200~300자 정도로 간결하게 작성해 주세요.
    설명은 반드시 **한글(가-힣)과 영어(a-z)만 사용하여 작성해야 합니다.**
    숫자, 특수문자, 한자는 포함할 수 없습니다.
    **검색해서 진위여부가 확실하게 검증된 답변만 작성하세요**
    """
    
    # LLM으로 답변 생성
    completion = client.chat.completions.create(
        model="qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=512,
        top_p=0.95
    )
    
    return completion.choices[0].message.content.strip()

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