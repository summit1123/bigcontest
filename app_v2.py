import json
import os
import re
import pandas as pd
import numpy as np
import typing_extensions as typing
import enum
from sys import argv
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import google.generativeai as genai 
import google.ai.generativelanguage as glm
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

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
    '현지인이용건수비중': 'LOCAL_UE_CNT_RAT',
    '남성회원수비중': 'RC_M12_MAL_CUS_CNT_RAT',
    '여성회원수비중': 'RC_M12_FME_CUS_CNT_RAT',
    '20대이하회원수비중': 'RC_M12_AGE_UND_20_CUS_CNT_RAT',
    '30대회원수비중': 'RC_M12_AGE_30_CUS_CNT_RAT',
    '40대회원수비중': 'RC_M12_AGE_40_CUS_CNT_RAT',
    '50대회원수비중': 'RC_M12_AGE_50_CUS_CNT_RAT',
    '60대이상회원수비중': 'RC_M12_AGE_OVR_60_CUS_CNT_RAT'
}

# JSON용 클래스

# 특정 요일 방문 횟수 비중을 나타내는 TypeDict
class VisitCountShare(typing.TypedDict):
    day: str          # 요일 (예: '월요일')
    visit_percentage: float  # 해당 요일의 방문 비중
    

class Category(enum.Enum):
    NONE = "None"
    Homestyle = "가정식"
    Chinese = "중식"
    SingleMenu = "단품요리 전문"
    Coffee = "커피"
    IceCream = "아이스크림/빙수"
    Pizza = "피자"
    Western = "양식"
    Chicken = "치킨"
    Japanese = "일식"
    BeerAndPub = "맥주/요리주점"
    Sandwich = "샌드위치/토스트"
    LunchBox = "도시락"
    Bakery = "베이커리"
    KoreanSnacks = "분식"
    Skewers = "꼬치구이"
    Tea = "차"
    Steak = "스테이크"
    Cafeteria = "구내식당/푸드코트"
    AsianIndian = "동남아/인도음식"
    Hamburger = "햄버거"
    Ricecake = "떡/한과"
    FoodTruck = "포장마차"
    Juice = "주스"
    TraditionalPub = "민속주점"
    Buffet = "부페"
    WorldCuisine = "기타세계요리"
    Donut = "도너츠"
    TransportCafe = "기사식당"
    NightSnack = "야식"
    FamilyRestaurant = "패밀리 레스토랑"
    

class orderType(enum.Enum):
    NONE = "None"
    highest = "highest"
    lowest = "lowest"
    
class filterType(enum.Enum):
    NONE = "None"
    Mon = "월요일이용비중"
    Tue = "화요일이용비중"
    Wed = "수요일이용비중"
    Thu = "목요일이용비중"
    Fri = "금요일이용비중"
    Sat = "토요일이용비중"
    Sun = "일요일이용비중"
    HR_5_11 = "5시11시이용건수비중"
    HR_12_13 = "12시13시이용건수비중"
    HR_14_17 = "14시17시이용건수비중"
    HR_18_22 = "18시22시이용건수비중"
    HR_23_4 = "23시4시이용건수비중"
    Local = "현지인이용건수비중"
    Mal = "남성회원수비중"
    Fme = "여성회원수비중"
    Age_20_Und = "20대이하회원수비중"
    Age_30 = "30대회원수비중"
    Age_40 = "40대회원수비중"
    Age_50 = "50대회원수비중"
    Age_60_Ovr = "60대이상회원수비중"
    
# 필터와 정렬 정보를 위한 TypedDict
class FilterOrder(typing.TypedDict):
    filter_type: filterType  # 필터 종류 (예: 요일, 성별 등)
    order_type: orderType   # 정렬 타입 (예: 'highest', 'lowest')

class Query(typing.TypedDict):
    is_recommend: typing.Required[bool]
    address: str
    category: Category
    Usage_Count_Range: str
    Spending_Amount_Range: str
    Average_Spending_Amount_Range: str
    # Visit_count_specific: VisitCountShare
    # Local_Visitor_Proportion: float
    ranking_condition: FilterOrder
    


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
    # csv_file_path = "final_coordinates.csv"
    csv_file_path = "final_restaurant_data.csv"
    df = pd.read_csv(os.path.join('./data', csv_file_path),encoding='cp949')
    return df

@st.cache_resource
def load_index():
    emb_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    index = faiss.read_index("store_index_2.faiss")
    name_index = faiss.read_index("name_index_2.faiss")
    nearby_index = faiss.read_index("store_nearby_index_2.faiss")
    # category_index = faiss.read_index("category_index.faiss")
    print("index loaded")
    return emb_model, index, nearby_index, name_index

# 추천용 함수

def search_by_location(region: str, type: str, preference: str=None):
    """특정 지역(예: 제주시 한림읍, 제주공항)의 특정 업종(예: 카페)인 식당목록을 반환합니다. 선호하는것이 있다면 적용합니다.

    Args:
        region (str): 지역명(예. 제주시 한림읍, 제주공항)
        type (str): 업종(예. 카페)
        preference (str, optional) 선호 사항(예. 전망(바다, 산, 일몰), 인테리어(모던, 고급스러운, 로맨틱한, 전통적인), 맛(감칠맛, 매운맛), 계절메뉴, 수제, 제주산 식재료)

    Returns:
        array: 조건에 맞는 식당 목록이 담긴 배열
    """
    # 예시 데이터 반환 (추후 실제 데이터로 교체)
    print(f"Call: search_by_location region:{region}, type:{type}, preference:{preference}")
    query_embedding = embed_model.encode([f"{region} {type}"])
    D, I = index.search(query_embedding, k=20)
    
    # 인덱스 결과에서 추출한 식당 데이터
    results = [df.iloc[i] for i in I[0]]

    # 선호 사항이 있는 경우 선호 사항과 Category 열의 유사도를 비교
    if preference:
        prefer_embedding = embed_model.encode(preference)

        # 각 식당의 Category 임베딩과 prefer_embedding 간의 유사도 계산
        similarities = []
        for result in results:
            category_embedding = embed_model.encode(f"{result['VIEW']} {result['INTERIOR']} {result['TASTE']} {result['SPECIALTY']}")
            similarity = np.dot(prefer_embedding, category_embedding)  # 코사인 유사도
            similarities.append((result, similarity))

        # 유사도가 높은 순으로 정렬하여 상위 5개 선택
        top_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
        results = [res[0] for res in top_results]

    # 선택된 결과를 문자열 형식으로 변환
    string_results = [
        f"이름: {res['MCT_NM']}, 업종: {res['MCT_TYPE']}, 주소: {res['ADDR']}, 이용건수구간: 상위 {str(res['UE_CNT_GRP'][2:]).replace('~', '-')}, 리뷰요약: {res['REVIEW_SUMMARY']}, 가게소개: {res['DESC_SUMMARY']}, 분류: {result['VIEW']} {result['INTERIOR']} {result['TASTE']} {result['SPECIALTY']}"
        for res in results
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
    # not_space = place.replace(" ", "")
    print(f"Call: search_nearby_location place:{place}")
    query_embedding = embed_model.encode([f"{place}"])
    D, I = name_index.search(query_embedding, k=5)
    # print(D)
    place_data = df.iloc[I[0]]
    # 후 처리. 1. 부분집합 2. Levenshtein
    closest_idx, similarity_score = find_closest_name(place, place_data['MCT_NM'].values)
    # closest_idx = find_closest_name(not_space, place_data['MCT_NM'].values)
    # print(similarity_score, place_data.iloc[closest_idx])
    print("장소 지정", place_data.iloc[closest_idx]['MCT_NM'], place_data.iloc[closest_idx]['ADDR'])
    place_location = np.array([[place_data.iloc[closest_idx]['Latitude'], place_data.iloc[closest_idx]['Longitude']]], dtype='float32')  # 사용자의 위도, 경도
    k = 30
    distances, indices = nearby_index.search(place_location, k)
    results = [
        f"이름: {df.iloc[indices[0][i]]['MCT_NM']}, 업종: {df.iloc[indices[0][i]]['MCT_TYPE']}, 주소: {df.iloc[indices[0][i]]['ADDR']}, 이용건수구간: 상위 {str(df.iloc[i]['UE_CNT_GRP'][2:]).replace('~', '-')}"
        for i in range(1, len(indices[0])) # 0은 자기자신
    ]
    print(results)
    return results

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
    
    output = None
    if 0 <= percentage <= 10:
        output = '1_상위 10% 이하'
    elif 10 < percentage <= 25:
        output = '2_10~25%'
    elif 25 < percentage <= 50:
        output = '3_25~50%'
    elif 50 < percentage <= 75:
        output = '4_50~75%'
    elif 75 < percentage <= 90:
        output =  '5_75~90%'
    elif 90 < percentage <= 100:
        output = '6_90% 초과(하위 10% 이하)'
    print(f"{raw_percentage}->{output}")
    return output

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
    
    
# 함수 호출용 아님
def filtering(dic):
    """일반 검색 대응

    Args:
        dic (dict): Json에서 가공된 Dictionary 데이터

    Returns:
        output (str): 출력할 텍스트
    """
    # 일반 쿼리 대응
    addr = dic.get("address", None)
    mct_type = dic.get("category", None)
    ranking_condition = dic.get('ranking_condition', None)
    filter_type = ranking_condition.get('filter_type', None)
    order_type = ranking_condition.get('order_type', None)
    
    # 형식 변환
    usage_Count_Range = map_to_group(dic.get("Usage_Count_Range", None))
    spending_Amount_Range = map_to_group(dic.get("Spending_Amount_Range", None))
    average_Spending_Amount_Range = map_to_group(dic.get("Average_Spending_Amount_Range", None))
    
    # 출력용 텍스트
    output_conditions = ""
    
    # 필터링
    conditions = []
    if addr is not None:
        conditions.append(df['ADDR'].str.contains(addr, na=False))
        output_conditions += f"주소: {addr}<br>"
    if mct_type is not None:
        conditions.append(df['MCT_TYPE'].str.contains(mct_type, na=False))
        output_conditions += f"업종: {mct_type}<br>"
    if usage_Count_Range is not None:
        conditions.append(df['UE_CNT_GRP'] == usage_Count_Range)
        output_conditions += f"이용건수: {usage_Count_Range[2:]}<br>"
    if spending_Amount_Range is not None:
        conditions.append(df['UE_AMT_GRP'] == spending_Amount_Range)
        output_conditions += f"이용금액: {spending_Amount_Range[2:]}<br>"
    if average_Spending_Amount_Range is not None:
        conditions.append(df['UE_AMT_PER_TRSN_GRP'] == average_Spending_Amount_Range)
        output_conditions += f"건당평균이용금액: {average_Spending_Amount_Range[2:]}<br>"
        
    if conditions:
        filtered_df = df.loc[pd.concat(conditions, axis=1).all(axis=1)]
        print("1차 필터링", len(filtered_df))
    else:
        return "대답할 수 없는 질문입니다.."
    
    if filter_type != "None" and order_type != "None":
        is_ascending = (order_type == "lowest")
        filtered_df = filtered_df.sort_values(by=matching_keys.get(filter_type, None), ascending=is_ascending)
        asending_text = "오름차순" if is_ascending else "내림차순"
        output_conditions += f"정렬조건: {filter_type} {asending_text}"
    
    output = None
    if len(filtered_df) == 0:
        print("조건", output_conditions)
        output = "검색 결과가 없습니다."
    else:
        # 조건 출력
        output = f"### 조건에 해당하는 식당을 찾았습니다.\n**식당명**: {filtered_df.iloc[0]['MCT_NM']}<br>**주소**: {filtered_df.iloc[0]['ADDR']}<hr>##### **조건**<br>{output_conditions}"
    
    return output

    
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

def merge_dicts(dicts):
    merged = {}

    for d in dicts:
        for key, value in d.items():
            if key == "ranking_condition" and key in merged:
                # ranking_condition의 filter_type이 None인 경우 넘어가고, 아니면 덮어씀
                if not value.get("filter_type"):
                    continue
                if value["filter_type"] != "None":
                    merged[key].update(value)
            else:
                # 다른 경우는 그냥 덮어씀
                merged[key] = value

    return merged


# Function Repository 등록
function_repository = {
    "search_by_location": search_by_location,
    "search_nearby_location": search_nearby_location,
    "check_user_options": check_user_options,
    "display_user_option_menu": display_user_option_menu,
    "change_user_option": change_user_option,
    
}

@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safe)
    print("model loaded...")
    return model

# 모델, 인덱스 로드
model = load_model()
embed_model, index, nearby_index, name_index = load_index()
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
if 'history' not in st.session_state:
    st.session_state.history = []
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

        # 선택한 옵션을 채팅 기록에 추가
        # st.session_state.chat_history.append({"role": "assistant", "content": f"선택된 나이대: {option1}, 방문 시간: {option2}, 현지인 기준: {option3}"})
        try:
            st.session_state.form_submitted = True
            st.session_state.options = [option1, option2, option3]
            print(st.session_state.options)
            st.rerun()
        except Exception as e:
            print(f"오류 발생: {str(e)}")
        
    if cancel_button:
        st.session_state.form_submitted = True
        # st.session_state.chat_history.append({"role": "assistant", "content": f"옵션 생략"})
        st.rerun()

user_input = None
    

if st.session_state.form_submitted:
    # 채팅 입력 받기
    user_input = st.chat_input("메시지를 입력하세요")   
    print(st.session_state.options)

for content in st.session_state.history:
    # print(content)
    with st.chat_message("assistant" if content['role'] == "model" else "user"):
        output = content['parts'][0]['text']
        if content['role'] == "user":
            output = output[output.find("질문:")+3:]
        st.markdown(output, unsafe_allow_html=True)

if user_input:
    st.session_state.history.append({
        'role': 'user',
        'parts': [{'text': user_input}]
    })
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        json_prompt = f"""질문에서 요구사항을 보고 JSON의 모든 항목(is_recommend, 주소, 업종, 이용건수구간, 이용금액구간, 건당평균이용금액구간)을 반드시 반환하라\n
        각 필드의 대한 설명이다. address:주소(예. 제주시 ㅁㅁ읍),    category:업종,   Usage_Count_Range:이용건수구간(예. 이용건수 상위 N%),  Spending_Amount_Range:이용금액구간(예. 이용금액구간 상위 N%),
        Average_Spending_Amount_Range:건당평균이용금액구간(예. 건당평균이용금액 상위 N%), is_recommend:(추천 혹은 연계되는 질문일경우(True. 예시: 추천해줘), 조건에 따른 검색일경우(False. 예시:조건이 XX하고, 가장 XX한것은?))\n
        ranking_condition는 없을 수도 있으며, 오직 순위를 나타내는 조건(가장 큰것, 가장 작은것)에만 해당한다. 
        \n질문: {user_input}"""
        
        recommend_prompt = f"""너는 제주도의 맛집을 추천해주는 사람이야.
        사용자의 질문에서 키워드를 찾고 함수를 사용하여 필요한 것을 찾아. 가정하지말고, 모르는걸 말하지마.\n
        이용건수구간이 작을수록 맛집일 가능성이 높아(예. 상위 10%는 상위 80% 보다 맛집일 거야\n
        가게의 리뷰요약과 가게소개를 보고 추천하면서 설명해줘\n
        질문: {user_input}"""
        
        print("입력", json_prompt)
        print("프롬프트 토큰", model.count_tokens(json_prompt))
        response = model.generate_content(json_prompt, 
            generation_config=genai.GenerationConfig(response_mime_type="application/json", response_schema=list[Query]),
            tools=[]) 
        
        dic = merge_dicts(json.loads(response.parts[0].text))
        print("JSON 데이터", dic)

        # is_recommend 로 추천인지 검색인지 확인. False(검색)일 경우 정확도를 위해 한번 더 검색 후 선택
        if not dic.get("is_recommend"):
            response2 = model.generate_content(json_prompt, 
            generation_config=genai.GenerationConfig(response_mime_type="application/json", response_schema=list[Query]),
            tools=[]) 
            dic2 = merge_dicts(json.loads(response2.parts[0].text))
            # 조건 체크 (만약 처음엔 false였는데 갑자기 true)
            if dic2.get("is_recommend"):
                print("원래 검색이였는데, 다음은 추천으로 인식")
            correct_dic = dic if len(dic) > len(dic2) else dic2
            print(f"2개 중 선택\n{dic}\n{dic2}\n->{correct_dic}")
            output = filtering(correct_dic)
            print("검색 출력", output)
            st.session_state.history.append({
                'role': 'model',
                'parts': [{'text': output}]
            })
            st.markdown(output, unsafe_allow_html=True) # 이런식으로 직접 넣는거면, history를 통해서 채팅 재구성은 불가능하다. 직접 로그를 구성해야
        # 추천방식. 정확도가 필요없다. 아마 여기에 나이대, 성별 가중치를 적용할것(프롬프트로?)
        else:
            print("추천 출력", recommend_prompt)
            output = st.session_state.chat_session.send_message(recommend_prompt,
                generation_config=genai.GenerationConfig(response_mime_type="text/plain", response_schema=None, temperature=0.1), tools=function_repository.values()) 
            st.markdown(output.text)
            st.session_state.history.append({
                'role': 'model',
                'parts': [{'text': output.text}]
            })
        print('-'*50)
        
# 제주시 한림읍에 있는 카페 목록이 필요해
# 제주시청역 근처 중국집 추천해줄래?
# 돈까스먹고싶은데 만원과 3만원정도 쓸건데 추천해주라
#streamlit run .\app_v2.py --logger.level=debug