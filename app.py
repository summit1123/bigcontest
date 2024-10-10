import json
import os
import re
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

# column_keys = ['기준연월','개설일자', '가맹점명', '업종',
#              '주소', '이용건수구간',
#              '이용금액구간', '건당평균이용금액구간',
#              '월요일이용비중', '화요일이용비중',
#              '수요일이용비중', '목요일이용비중',
#              '금요일이용비중', '토요일이용비중',
#              '일요일이용비중', '5시~11시이용비중',
#              '12시~13시이용비중', '14시~17시이용비중',
#              '18시~22시이용비중', '23시~4시이용비중',
#              '현지인이용비중', '남성회원수비중',
#              '여성회원수비중', '20대이하회원수비중',
#              '30대회원수비중', '40대회원수비중',
#              '50대회원수비중', '60대이상회원수비중']

orderby_keys = ['월요일이용비중', '화요일이용비중',
             '수요일이용비중', '목요일이용비중',
             '금요일이용비중', '토요일이용비중',
             '일요일이용비중', '5시~11시이용비중',
             '12시~13시이용비중', '14시~17시이용비중',
             '18시~22시이용비중', '23시~4시이용비중',
             '현지인이용비중', '남성회원수비중',
             '여성회원수비중', '20대이하회원수비중',
             '30대회원수비중', '40대회원수비중',
             '50대회원수비중', '60대이상회원수비중']

matching_keys = {
    '월요일이용비중': 'MON_UE_CNT_RAT',
    '화요일이용비중': 'TUE_UE_CNT_RAT',
    '수요일이용비중': 'WED_UE_CNT_RAT',
    '목요일이용비중': 'THU_UE_CNT_RAT',
    '금요일이용비중': 'FRI_UE_CNT_RAT',
    '토요일이용비중': 'SAT_UE_CNT_RAT',
    '일요일이용비중': 'SUN_UE_CNT_RAT',
    '5시~11시이용비중': 'HR_5_11_UE_CNT_RAT',
    '12시~13시이용비중': 'HR_12_13_UE_CNT_RAT',
    '14시~17시이용비중': 'HR_14_17_UE_CNT_RAT',
    '18시~22시이용비중': 'HR_18_22_UE_CNT_RAT',
    '23시~4시이용비중': 'HR_23_4_UE_CNT_RAT',
    '현지인이용비중': 'LOCAL_UE_CNT_RAT',
    '남성회원수비중': 'RC_M12_MAL_CUS_CNT_RAT',
    '여성회원수비중': 'RC_M12_FME_CUS_CNT_RAT',
    '20대이하회원수비중': 'RC_M12_AGE_UND_20_CUS_CNT_RAT',
    '30대회원수비중': 'RC_M12_AGE_30_CUS_CNT_RAT',
    '40대회원수비중': 'RC_M12_AGE_40_CUS_CNT_RAT',
    '50대회원수비중': 'RC_M12_AGE_50_CUS_CNT_RAT',
    '60대이상회원수비중': 'RC_M12_AGE_OVR_60_CUS_CNT_RAT'
}

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
st.set_page_config(page_title="🍊제주 맛집 추천")
# 함수 정의

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
    column_embeddings = emb_model.encode(orderby_keys)
    print("index loaded")
    return emb_model, index, nearby_index, column_embeddings


def search_by_location(region: str, type: str):
    """특정 지역(예: 제주시 한림읍, 제주공항)의 특정 업종(예: 카페)인 식당목록을 반환합니다. 조건:정렬이 필요없을것

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
        f"이름: {df.iloc[i]['MCT_NM']}, 업종: {df.iloc[i]['MCT_TYPE']}, 주소: {df.iloc[i]['ADDR']}"
        for i in I[0]
    ]
    print(string_results)
    return string_results

def search_nearby_location(place: str):
    """특정 장소(예: 음식점, 관광지 이름)의 주변 식당에 대한 정보를 제공합니다.

    Args:
        place (str): 식당 및 장소명(예. 제제김밥, 제주공항)
        
    Returns:
        array: 조건에 맞는 식당 목록이 담긴 배열
    """
    not_space = place.replace(" ", "")
    print(f"Call: search_nearby_location place:{place}->{not_space}")
    query_embedding = embed_model.encode([f"{not_space}"])
    D, I = index.search(query_embedding, k=5)
    # print(D)
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

# 단순 정보 검색용(대량)
def query_with_condition(addr:str=None, mct_type:str=None, visit_range:str=None, total_spending_range:str=None, average_spending_range:str=None, sortby:str=None, is_ascending:bool=False):
    f"""조건에 맞는 가게를 찾습니다. 인수는 선택적으로 사용됩니다.

    Args:
        addr (str, optional): 주소(예. 서귀포시 대정읍)
        mct_type (str, optional): 업종(예. 가정식, 단품요리 전문)
        visit_range (str, optional): 이용건수구간(예. 상위 10%)
        total_spending_range (str, optional): 이용금액구간(예. 상위 20%)
        average_spending_range (str, optional): 평균이용금액구간(예. 10~25%)
        sortby (str): 외부 정렬 기준. 예) '월요일이용비중', '30대이용비중', '현지인이용비중'
        is_ascending (bool): 오름차순 여부.

    Returns:
        List: 조건에 맞는 식당 목록
    """
    print(f"Call query_restaurant_with_condition")
    print("Function arguments:", locals())
    
    # 형식 변환
    visit_range = map_to_group(visit_range)
    total_spending_range = map_to_group(total_spending_range)
    average_spending_range = map_to_group(average_spending_range)
    
    # 필터링
    conditions = []
    if addr is not None:
        conditions.append(df['ADDR'].str.contains(addr, na=False))
    if mct_type is not None:
        conditions.append(df['MCT_TYPE'].str.contains(mct_type, na=False))
    if visit_range is not None:
        conditions.append(df['UE_CNT_GRP'] == visit_range)
    if total_spending_range is not None:
        conditions.append(df['UE_AMT_GRP'] == total_spending_range)
    if average_spending_range is not None:
        conditions.append(df['UE_AMT_PER_TRSN_GRP'] == average_spending_range)
        

    if conditions:
        filtered_df = df.loc[pd.concat(conditions, axis=1).all(axis=1)]
        print(filtered_df)
    
    # 기준 임베딩
    user_input_embedding = embed_model.encode([sortby])
    # 5. 코사인 유사도 계산
    cosine_similarities = np.dot(column_embeddings, user_input_embedding.T).flatten()

    # 6. 가장 유사한 컬럼명 선택
    best_match_index = np.argmax(cosine_similarities)
    selected_column = orderby_keys[best_match_index]
    
    sorted_df = filtered_df.sort_values(by=matching_keys.get(selected_column, None), ascending=is_ascending)

    top_5 = sorted_df.head(5)
    print("정렬조건", selected_column)
    result_list = []
    for index, row in top_5.iterrows():
        result_list.append(f"가게명: {row['MCT_NM']}, 주소: {row['ADDR']}, 업종: {row['MCT_TYPE']}, 이용건수: {row['UE_CNT_GRP']}, 현지인비중: {row['LOCAL_UE_CNT_RAT']}")
    print(result_list)
    
    return result_list

# 상위 퍼센트 값(예: 14%, 79% 등)을 주어진 구간에 맞게 변환하는 함수
def map_to_group(raw_percentage):
    """
    사용자의 입력 퍼센티지 값을  구간으로 변환
    Args:
        percentage (float): 상위 퍼센티지 값 (예: 14.5, 79 등)

    Returns:
        str: 변환된 구간 값
    """
    if raw_percentage is None:
        return None
    
    pattern = r"(?:상위\s*)?([\d\.]+)%"
    match = re.search(pattern, raw_percentage)
    if match:
        percentage = float(match.group(1))
    else:
        return None
    
    if 0 <= percentage < 10:
        return '1_상위10%이하'
    elif 10 <= percentage < 25:
        return '2_10~25%'
    elif 25 <= percentage < 50:
        return '3_25~50%'
    elif 50 <= percentage < 75:
        return '4_50~75%'
    elif 75 <= percentage < 90:
        return '5_75~90%'
    elif 90 <= percentage <= 100:
        return '6_90% 초과(하위 10% 이하)'
    else:
        return None  # 퍼센티지 값이 범위를 벗어날 때


# 옵션 확인용 함수
def check_user_options():
    """사용자가 설정한 '나이대', '방문시간', '현지인식당여부' 정보를 가져옵니다. (사용자는 처음에 별도의 UI를 통해 3가지를 설정하고 시작합니다)

    Returns:
        List: [나이대, 방문시간, 현지인식당여부]가 담긴 리스트
    """
    print(f"Call: check_user_options")
    return st.session_state.options

def change_user_option(opt1:str=None, opt2:str=None, opt3:bool=None):
    """나이대, 방문시간, 현지인식당여부 정보를 일부 혹은 전부 변경합니다.

    Args:
        opt1 (str, optional): 나이대(["20대 이하", "30대", "40대", "50대", "60대 이상"] 중 하나)
        opt2 (str, optional): 방문시간(["오전(5시~11시)", "점심(12시~13시)", "오후(14시~17시)", "저녁(18시~22시)", "심야(23시~4시)"] 중 하나)
        opt3 (bool, optional): 현지인식당여부(True, False)
    """
    if opt1:
        st.session_state.options[0] = opt1
    if opt2:
        st.session_state.options[1] = opt2
    if opt3 != None:
        st.session_state.options[2] = opt3
    print("옵션 변경",opt1, opt2, opt3)
    

def display_user_option_menu():
    """사용자가 다시 옵션(나이대, 방문시간, 현지인식당여부)을 설정하도록 메뉴를 표시합니다. 직접 선택메뉴를 생성하기때문에 "알겠다"는 정도의 대답이외에는 필요없습니다.

    Returns:
        None: return 값이 없습니다.
    """
    print("다시 메뉴 표시")
    st.session_state.form_submitted = False
    st.rerun()


# 미사용 함수들
# def recommend_by_average_spending(type: str, region:str=None, min_spending:str=None, max_spending:str=None):
#     """특정 업종(예: 한식당)의 건당 평균 이용금액이 특정 금액 범위에 해당하는 곳을 추천합니다.

#     Args:
#         type (str):  예) "한식당"
#         region (str, optional): (선택) 특정 지역 필터링 가능.. Defaults to None.
#         min_spending (str, optional): 최저 이용 금액. Defaults to None.
#         max_spending (str, optional): 최고 이용 금액. Defaults to None.

#     Returns:
#         array: 조건에 맞는 식당 목록이 담긴 배열
#     """
#     print(f"Call: recommend_by_average_spending region:{region}, type:{type}, min_spending:{min_spending}, max_spending:{max_spending}")
#     if not region:
#         region = "제주공항"
#     return [
#         {"store_name": f"{type}1", "store_address": f"{region} 중심가", "category": type},
#         {"store_name": f"{type}2", "store_address": f"{region} 남쪽", "category": type},
#     ]

# # 특정 조건에 따라 데이터를 검색하는 함수
# def search_by_conditions(region: str, business_type: str, sort_by: str, top_n: int = 1):
#     """특정 조건에 따라 식당을 검색합니다.
    
#     Args:
#         region (str): 지역명 (예. 제주시 한림읍)
#         business_type (str): 업종명 (예. 카페)
#         sort_by (str): 정렬 기준 (예. "현지인 이용 비중", "30대 이용 비중")
#         top_n (int): 상위 몇 개의 가맹점을 반환할지 (기본값 1)
        
#     Returns:
#         list: 조건에 맞는 상점 리스트
#     """
    
    
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
    "query_with_condition": query_with_condition,
    "search_by_location": search_by_location,
    "search_nearby_location": search_nearby_location,
    "check_user_options": check_user_options,
    "display_user_option_menu": display_user_option_menu,
    "change_user_option": change_user_option,
    
}

# instructions = "너는 제주도의 맛집을 추천해주는 사람이야. 사용자의 질문에서 키워드를 찾고 함수를 호출하여 필요한 것을 찾아. 가정하지말고, 모르는걸 말하지마. 추천 이유도 설명해"

@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safe, tools=function_repository.values())
    print("model loaded...")
    return model

# 모델, 인덱스 로드
model = load_model()
embed_model, index, nearby_index, column_embeddings = load_index()
df = load_data()
history = []

# Streamlit 시작

# 세션 상태 초기화
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False  # 폼 제출 상태 확인
if "system_message_displayed" not in st.session_state:
    st.session_state.system_message_displayed = False
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []  # 채팅 내역 저장
if "options" not in st.session_state:
    st.session_state.options = [None, None, None]

if "chat_session" not in st.session_state:    
    st.session_state["chat_session"] = model.start_chat(history=[], enable_automatic_function_calling=True) 



st.title("반갑다!👋")
st.subheader("제주 맛집을 추천해주겠다")
st.write("")

if not st.session_state.form_submitted:
    with st.form(key="user_options_form"):
        st.write("너에게 맞는 추천을 제공하기위해 정보가 필요하다")
        st.write("")
        st.write("맞춤형 추천을 위해 선택하라")
        option1 = st.selectbox("나이대", ["20대 이하", "30대", "40대", "50대", "60대 이상"])
        option2 = st.selectbox("방문 시간", ["오전(5시~11시)", "점심(12시~13시)", "오후(14시~17시)", "저녁(18시~22시)", "심야(23시~4시)"])
        option3 = st.checkbox("현지인 맛집", value=False)

        col1, col2 = st.columns(2)  # 2개의 열 생성

        with col1:
            submit_button = st.form_submit_button(label="완료")  # 완료 버튼

        with col2:
            cancel_button = st.form_submit_button(label="생략")  # 생략 버튼 (폼 제출하지 않음)
    
    if submit_button:
        st.session_state.form_submitted = True
        st.session_state.options = [option1, option2, option3]
        # 선택한 옵션을 채팅 기록에 추가
        # st.session_state.chat_history.append({"role": "assistant", "content": f"선택된 나이대: {option1}, 방문 시간: {option2}, 현지인 기준: {option3}"})
        st.rerun()
        
    if cancel_button:
        st.session_state.form_submitted = True
        # st.session_state.chat_history.append({"role": "assistant", "content": f"옵션 생략"})
        st.rerun()

user_input = None

# 시스템 메시지
# if not st.session_state.system_message_displayed and st.session_state.form_submitted:
#     system_message = "반갑다! 무엇을 도와줄까? 질문을 입력하라."
#     # 현재 방식에 맞게 바꿔야 함
#     # st.session_state.chat_history.append({"role": "assistant", "content": system_message})
#     st.session_state.system_message_displayed = True
    

if st.session_state.form_submitted:
    # 채팅 입력 받기
    user_input = st.chat_input("메시지를 입력하세요")   

for content in st.session_state.chat_session.history:
    # print(content)
    if (len(content.parts) > 1) or not content.parts[0].text:
            continue
    with st.chat_message("assistant" if content.role == "model" else "user"):
        output = content.parts[0].text
        if content.role == "user":
            output = output[output.find("질문:")+3:]
        st.markdown(output)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        edited_prompt = f"""너는 제주도의 맛집을 추천해주는 사람이야.
        사용자의 질문에서 키워드를 찾고 함수를 사용하여 필요한 것을 찾아. 가정하지말고, 모르는걸 말하지마.
        함수를 사용할때, 함수의 'Description', 'Arg', 'Returns'을 출력하지마.
        단순 정보 검색성 질문에는 "query_with_condition" 함수를 사용하고,
        추천형 질문의 경우에는 그 외의 함수를 사용해\n
        질문: {user_input}"""
        print("프롬프트 토큰", model.count_tokens(edited_prompt))
        response = st.session_state.chat_session.send_message(edited_prompt)  
        print("토큰\n", response.usage_metadata)
        st.markdown(response.text)
        
# 제주시 한림읍에 있는 카페 목록이 필요해
# 제주시청역 근처 중국집 추천해줄래?
# 돈까스먹고싶은데 만원과 3만원정도 쓸건데 추천해주라