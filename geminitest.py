import json
import os
import pandas as pd
import numpy as np
from sys import argv
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import google.generativeai as genai 
import google.ai.generativelanguage as glm
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance


with open('./secrets.json') as f:
    secrets = json.loads(f.read())
GOOGLE_API_KEY = secrets['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

@st.cache_resource
def load_data():
    # CSV 로드
    csv_file_path = "final_coordinates.csv"
    df = pd.read_csv(os.path.join('./data', csv_file_path),encoding='cp949')
    return df

@st.cache_resource
def load_index():
    emb_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    index = faiss.read_index("store_index.faiss")
    nearby_index = faiss.read_index("store_nearby_index.faiss")
    print("index loaded")
    return emb_model, index, nearby_index


st.title("제주맛zip")


def search_by_location(region: str, type: str):
    """특정 지역(예: 제주시 한림읍, 제주공항)의 특정 업종(예: 카페)인 식당목록을 반환합니다.

    Args:
        region (str): 지역명(예. 제주시 한림읍, 제주공항)
        type (str): 업종(예. 카페)

    Returns:
        array: 조건에 맞는 식당 목록이 담긴 배열
    """
    # 예시 데이터 반환 (추후 실제 데이터로 교체)
    print(f"Call: search_by_location region:{region}, type:{type}")
    query_embedding = embed_model.encode([f"{region} {type}"])
    D, I = index.search(query_embedding, k=10)
    # result = df.iloc[I[0]]  # 검색된 인덱스 번호와 원본 데이터프레임 연결
    string_results = [
        f"이름: {df.iloc[i]['MCT_NM']}, 업종: {df.iloc[i]['MCT_TYPE']}, 주소: {df.iloc[i]['ADDR'], }"
        for i in I[0]
    ]
    return string_results

def search_nearby_location(place: str):
    """특정 장소(예: 음식점, 관광지 이름)의 주변 식당에 대한 정보를 제공합니다.

    Args:
        place (str): 식당 및 장소명(예. 제제김밥, 제주공항)
        
    Returns:
        array: 조건에 맞는 식당 목록이 담긴 배열
    """
    not_space = place.replace(" ", "")
    print(f"Call: search_by_location place:{place}->{not_space}")
    query_embedding = embed_model.encode([f"{not_space}"])
    D, I = index.search(query_embedding, k=5)
    print(D)
    place_data = df.iloc[I[0]]
    # 후 처리. 1. 부분집합 2. Levenshtein
    closest_idx, similarity_score = find_closest_name(not_space, place_data['MCT_NM'].values)
    # closest_idx = find_closest_name(not_space, place_data['MCT_NM'].values)
    print(place_data)
    print(similarity_score, place_data.iloc[closest_idx])
    place_location = np.array([[place_data.iloc[closest_idx]['Latitude'], place_data.iloc[closest_idx]['Longitude']]], dtype='float32')  # 사용자의 위도, 경도
    k = 30
    distances, indices = nearby_index.search(place_location, k)
    results = [
        f"이름: {df.iloc[indices[0][i]]['MCT_NM']}, 업종: {df.iloc[indices[0][i]]['MCT_TYPE']}, 주소: {df.iloc[indices[0][i]]['ADDR']}"
        for i in range(1, len(indices[0])) # 0은 자기자신
    ]
    print(results)
    return results
    

def recommend_by_average_spending(type: str, region:str=None, min_spending:str=None, max_spending:str=None):
    """특정 업종(예: 한식당)의 건당 평균 이용금액이 특정 금액 범위에 해당하는 곳을 추천합니다.

    Args:
        type (str):  예) "한식당"
        region (str, optional): (선택) 특정 지역 필터링 가능.. Defaults to None.
        min_spending (str, optional): 최저 이용 금액. Defaults to None.
        max_spending (str, optional): 최고 이용 금액. Defaults to None.

    Returns:
        array: 조건에 맞는 식당 목록이 담긴 배열
    """
    print(f"Call: recommend_by_average_spending region:{region}, type:{type}, min_spending:{min_spending}, max_spending:{max_spending}")
    if not region:
        region = "제주공항"
    return [
        {"store_name": f"{type}1", "store_address": f"{region} 중심가", "category": type},
        {"store_name": f"{type}2", "store_address": f"{region} 남쪽", "category": type},
    ]

# 특정 조건에 따라 데이터를 검색하는 함수
def search_by_conditions(region: str, business_type: str, sort_by: str, top_n: int = 1):
    """특정 조건에 따라 식당을 검색합니다.
    
    Args:
        region (str): 지역명 (예. 제주시 한림읍)
        business_type (str): 업종명 (예. 카페)
        sort_by (str): 정렬 기준 (예. "현지인 이용 비중", "30대 이용 비중")
        top_n (int): 상위 몇 개의 가맹점을 반환할지 (기본값 1)
        
    Returns:
        list: 조건에 맞는 상점 리스트
    """
    

# def find_closest_name(query, candidates):
#     closest_index = -1
#     min_distance = float('inf')
    
#     for idx, candidate in enumerate(candidates):
#         dist = levenshtein_distance(query, candidate)
#         if dist < min_distance:
#             min_distance = dist
#             closest_index = idx
            
#     return closest_index

# Jaccard Similarity 계산 함수
def jaccard_similarity(str1, str2):
    # 각 문자열을 단어 집합으로 변환
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    # 교집합과 합집합 계산
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    # Jaccard Similarity 반환
    return intersection / union if union != 0 else 0

def find_closest_name(query, place_data):
    # 가장 유사한 이름과 유사도 저장
    closest_match = None
    highest_score = -1

    # Jaccard Similarity를 사용하여 가장 유사한 결과 찾기
    for index, name in enumerate(place_data):
        score = jaccard_similarity(query, name)
        if score > highest_score:
            highest_score = score
            closest_match = index

    return closest_match, highest_score




# Function Repository 등록
function_repository = {
    "search_by_location": search_by_location,
    "recommend_by_average_spending": recommend_by_average_spending,
    "search_nearby_location": search_nearby_location
}

instructions = "너는 제주도의 맛집을 추천해주는 사람이야. 사용자의 질문에서 키워드를 찾고 함수를 호출하여 필요한 것을 찾아. 가정하지말고, 모르는걸 말하지마. 추천 이유도 설명해"

@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safe, tools=function_repository.values())
    print("model loaded...")
    return model

model = load_model()
embed_model, index, nearby_index = load_index()
df = load_data()
history = []

if "chat_session" not in st.session_state:    
    st.session_state["chat_session"] = model.start_chat(history=history, enable_automatic_function_calling=True) 


for content in st.session_state.chat_session.history:
    if (len(content.parts) > 1) or not content.parts[0].text:
            continue
    with st.chat_message("assistant" if content.role == "model" else "user"):
        output = content.parts[0].text
        if content.role == "user":
            output = output[output.find("질문:")+3:]
        st.markdown(output)

if prompt := st.chat_input("메시지를 입력하세요."):    
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        edited_prompt = f"너는 제주도의 맛집을 추천해주는 사람이야. 사용자의 질문에서 키워드를 찾고 함수를 호출하여 필요한 것을 찾아. 가정하지말고, 모르는걸 말하지마. 추천 이유도 설명해\n질문: {prompt}"
        
        response = st.session_state.chat_session.send_message(edited_prompt)  
        st.markdown(response.text)
        
# 제주시 한림읍에 있는 카페 목록이 필요해
# 제주시청역 근처 중국집 추천해줄래?
# 돈까스먹고싶은데 만원과 3만원정도 쓸건데 추천해주라