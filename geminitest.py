import json
import os
from sys import argv
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import google.generativeai as genai 
import google.ai.generativelanguage as glm
import streamlit as st


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


st.title("Gemini-Bot")


# Streamlit 캐싱

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
    return [
        {"store_name": "식당1", "store_address": f"{region} 중심가", "category": type},
        {"store_name": "식당2", "store_address": f"{region} 남쪽", "category": type},
    ]

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
    """특정 조건에 따라 상점을 검색합니다.
    
    Args:
        region (str): 지역명 (예. 제주시 한림읍)
        business_type (str): 업종명 (예. 카페)
        sort_by (str): 정렬 기준 (예. "현지인 이용 비중", "30대 이용 비중")
        top_n (int): 상위 몇 개의 가맹점을 반환할지 (기본값 1)
        
    Returns:
        list: 조건에 맞는 상점 리스트
    """



# Function Repository 등록
function_repository = {
    "search_by_location": search_by_location,
    "recommend_by_average_spending": recommend_by_average_spending
}

instructions = "너는 제주도의 맛집을 추천해주는 사람이야. 사용자의 질문에서 키워드를 찾고 함수를 호출하여 필요한 것을 찾아. 가정과 예상, 역으로 사용자한테 찾으려 하지마"

@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safe, tools=function_repository.values(), system_instruction=instructions)
    print("model loaded...")
    return model

model = load_model()
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
        edited_prompt = f"너는 제주도의 맛집을 추천해주는 사람이야. 사용자의 질문에서 키워드를 찾고 함수를 호출하여 필요한 것을 찾아. 가정과 예상, 역으로 사용자한테 찾으려 하지마, 사용된 함수를 사용한 판단근거를 함께 서술하세요\n질문: {prompt}"
        
        response = st.session_state.chat_session.send_message(edited_prompt)  
        st.markdown(response.text)
        
# 제주시 한림읍에 있는 카페 목록이 필요해
# 제주시청역 근처 중국집 추천해줄래?
# 돈까스먹고싶은데 만원과 3만원정도 쓸건데 추천해주라