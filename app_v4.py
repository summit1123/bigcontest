import streamlit as st
st.set_page_config(
    page_title="제주맛.zip 🍊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)
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
import faiss
from sentence_transformers import SentenceTransformer

# API 키 설정
with open('./secrets.json') as f:
    secrets = json.loads(f.read())
GOOGLE_API_KEY = secrets['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

# Safety settings
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

# JSON용 클래스들
class VisitCountShare(typing.TypedDict):
    day: str          
    visit_percentage: float  

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

class FilterOrder(typing.TypedDict):
    filter_type: filterType  
    order_type: orderType   

class Query(typing.TypedDict):
    is_recommend: typing.Required[bool]
    address: str
    category: Category
    Usage_Count_Range: str
    Spending_Amount_Range: str
    Average_Spending_Amount_Range: str
    ranking_condition: FilterOrder

def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_file) as f:
        st.markdown(f"""<style>{f.read()}</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    csv_file_path = "final_restaurant_data_v2.csv"
    df = pd.read_csv(os.path.join('./data', csv_file_path), encoding='utf-8-sig')
    return df

@st.cache_resource
def load_index():
    emb_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    index = faiss.read_index("store_index_2.faiss")
    name_index = faiss.read_index("name_index_2.faiss")
    nearby_index = faiss.read_index("store_nearby_index_2.faiss")
    print("index loaded")
    return emb_model, index, nearby_index, name_index

@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safe)
    print("model loaded...")
    return model

# 유틸리티 함수들
def map_to_group(raw_percentage):
    if raw_percentage is None:
        return None
    
    pattern = r"(?:상위\s*)?([\d\.]+)%"
    match = re.search(pattern, raw_percentage)
    if match:
        percentage = float(match.group(1))
    else:
        return None
    
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

def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def find_closest_name(query, place_data):
    closest_match = None
    highest_score = -1
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
                if not value.get("filter_type"):
                    continue
                if value["filter_type"] != "None":
                    merged[key].update(value)
            else:
                merged[key] = value
    return merged

# 핵심 검색/추천 함수들
def search_by_location(region: str, type: str, preference: str=None):
    """특정 지역의 특정 업종 식당목록을 반환합니다."""
    print(f"Call: search_by_location region:{region}, type:{type}, preference:{preference}")
    query_embedding = embed_model.encode([f"{region} {type}"])
    D, I = index.search(query_embedding, k=20)
    
    results = [df.iloc[i] for i in I[0]]

    if preference:
        filtered_results = []
        for result in results:
            keywords = json.loads(result['keywords'])
            all_keywords = []
            for category_keywords in keywords.values():
                all_keywords.extend(category_keywords)
            
            if any(pref.lower() in " ".join(all_keywords).lower() for pref in preference.split()):
                filtered_results.append(result)
        
        results = filtered_results[:10] if filtered_results else results[:10]

    string_results = []
    for res in results:
        keywords = json.loads(res['keywords'])
        main_features = {
            '맛': keywords['TASTE'][:2] if keywords['TASTE'] else [],
            '특징': keywords['RESTAURANT_STYLE'][:2] if keywords['RESTAURANT_STYLE'] else [],
            '분위기': keywords['ATMOSPHERE'][:2] if keywords['ATMOSPHERE'] else []
        }
        
        feature_str = ", ".join([f"{k}: {', '.join(v)}" for k, v in main_features.items() if v])
        
        result_str = (
            f"이름: {res['MCT_NM']}, "
            f"업종: {res['MCT_TYPE']}, "
            f"주소: {res['ADDR']}, "
            f"이용건수구간: 상위 {str(res['UE_CNT_GRP'][2:]).replace('~', '-')}, "
            f"리뷰요약: {res['review_summary']}, "
            f"주요특성: [{feature_str}]"
        )
        string_results.append(result_str)
    
    return string_results

def search_nearby_location(place: str):
    """특정 장소 주변의 식당 정보를 제공합니다."""
    print(f"Call: search_nearby_location place:{place}")
    query_embedding = embed_model.encode([f"{place}"])
    D, I = name_index.search(query_embedding, k=5)
    place_data = df.iloc[I[0]]
    
    closest_idx, similarity_score = find_closest_name(place, place_data['MCT_NM'].values)
    print("장소 지정", place_data.iloc[closest_idx]['MCT_NM'], place_data.iloc[closest_idx]['ADDR'])
    
    place_location = np.array([[place_data.iloc[closest_idx]['Latitude'], 
                              place_data.iloc[closest_idx]['Longitude']]], dtype='float32')
    
    k = 30
    distances, indices = nearby_index.search(place_location, k)
    
    results = []
    for i in range(1, len(indices[0])):
        res = df.iloc[indices[0][i]]
        keywords = json.loads(res['keywords'])
        main_features = {
            '맛': keywords['TASTE'][:2] if keywords['TASTE'] else [],
            '특징': keywords['RESTAURANT_STYLE'][:2] if keywords['RESTAURANT_STYLE'] else [],
            '분위기': keywords['ATMOSPHERE'][:2] if keywords['ATMOSPHERE'] else []
        }
        feature_str = ", ".join([f"{k}: {', '.join(v)}" for k, v in main_features.items() if v])
        
        result_str = (
            f"이름: {res['MCT_NM']}, "
            f"업종: {res['MCT_TYPE']}, "
            f"주소: {res['ADDR']}, "
            f"이용건수구간: 상위 {str(res['UE_CNT_GRP'][2:]).replace('~', '-')}, "
            f"리뷰요약: {res['review_summary']}, "
            f"주요특성: [{feature_str}]"
        )
        results.append(result_str)
    
    return results

# 옵션 관련 함수들
def check_user_options():
    """사용자 설정 옵션을 반환합니다."""
    print(f"Call: check_user_options")
    return st.session_state.options

def change_user_option(opt1:str=None, opt2:str=None, opt3:bool=None):
    """옵션을 변경합니다."""
    if opt1:
        st.session_state.options[0] = opt1
    if opt2:
        st.session_state.options[1] = opt2
    if opt3 != None:
        st.session_state.options[2] = opt3
    print("옵션 변경",opt1, opt2, opt3)

def display_user_option_menu():
    """옵션 메뉴를 다시 표시합니다."""
    print("다시 메뉴 표시")
    st.session_state.form_submitted = False
    st.rerun()

def filtering(dic):
    """검색 조건에 따른 필터링을 수행합니다."""
    addr = dic.get("address", None)
    mct_type = dic.get("category", None)
    ranking_condition = dic.get('ranking_condition', None)
    filter_type = ranking_condition.get('filter_type', None)
    order_type = ranking_condition.get('order_type', None)
    
    usage_Count_Range = map_to_group(dic.get("Usage_Count_Range", None))
    spending_Amount_Range = map_to_group(dic.get("Spending_Amount_Range", None))
    average_Spending_Amount_Range = map_to_group(dic.get("Average_Spending_Amount_Range", None))
    
    output_conditions = ""
    
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
        keywords = json.loads(filtered_df.iloc[0]['keywords'])
        main_features = {
            '맛': keywords['TASTE'][:2] if keywords['TASTE'] else [],
            '특징': keywords['RESTAURANT_STYLE'][:2] if keywords['RESTAURANT_STYLE'] else [],
            '분위기': keywords['ATMOSPHERE'][:2] if keywords['ATMOSPHERE'] else []
        }
        feature_str = ", ".join([f"{k}: {', '.join(v)}" for k, v in main_features.items() if v])
        
        output = f"""### 조건에 해당하는 식당을 찾았습니다.
**식당명**: {filtered_df.iloc[0]['MCT_NM']}<br>
**주소**: {filtered_df.iloc[0]['ADDR']}<br>
**리뷰요약**: {filtered_df.iloc[0]['review_summary']}<br>
**주요특성**: [{feature_str}]<hr>
##### **검색 조건**<br>{output_conditions}"""
    
    return output

# Function Repository 등록
function_repository = {
    "search_by_location": search_by_location,
    "search_nearby_location": search_nearby_location,
    "check_user_options": check_user_options,
    "display_user_option_menu": display_user_option_menu,
    "change_user_option": change_user_option,
}

# 메인 실행
def main():
    load_css()

    # 메인 컨테이너
    with st.container():
        # 헤더 섹션
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
                <div class="title-container">
                    <span class="emoji-decoration">🍊</span>
                    <span class="main-title">제주맛.zip</span>
                    <span class="emoji-decoration">🌊</span>
                    <div class="sub-title">
                        제주도의 맛있는 발견
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # 옵션 폼
    if not st.session_state.form_submitted:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            with st.form("user_options_form"):
                st.markdown('<div class="form-title">🎯 맞춤 설정</div>', unsafe_allow_html=True)
                st.markdown('<div class="form-subtitle">더 정확한 추천을 위해 정보를 입력해주세요</div>', 
                          unsafe_allow_html=True)
                
                input_col1, input_col2, input_col3 = st.columns(3)
                
                with input_col1:
                    st.markdown('<div class="white-box"></div>', unsafe_allow_html=True)
                    option1 = st.selectbox(
                        "나이대",
                        ["20대 이하", "30대", "40대", "50대", "60대 이상"],
                        key="age_select"
                    )
                
                with input_col2:
                    st.markdown('<div class="white-box"></div>', unsafe_allow_html=True)
                    option2 = st.selectbox(
                        "방문 시간",
                        ["오전(5시~11시)", "점심(12시~13시)", "오후(14시~17시)", 
                         "저녁(18시~22시)", "심야(23시~4시)"],
                        key="time_select"
                    )
                
                with input_col3:
                    st.markdown('<div class="white-box"></div>', unsafe_allow_html=True)
                    option3 = st.checkbox("현지인 맛집", value=False)
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    submit = st.form_submit_button(
                        "완료",
                        use_container_width=True,
                    )
                with btn_col2:
                    skip = st.form_submit_button(
                        "생략",
                        use_container_width=True,
                    )

                if submit or skip:
                    if submit:
                        st.session_state.options = [option1, option2, option3]
                    st.session_state.form_submitted = True
                    st.rerun()

    # 채팅 인터페이스
    if st.session_state.form_submitted:
        st.markdown("### 💬 맛집 추천 채팅")
        with st.container():
            # 채팅 히스토리 표시
            for content in st.session_state.history:
                with st.chat_message("assistant" if content['role'] == "model" else "user"):
                    output = content['parts'][0]['text']
                    if content['role'] == "user":
                        output = output[output.find("질문:")+3:]
                    st.markdown(output, unsafe_allow_html=True)
            
            # 채팅 입력
            user_input = st.chat_input("어떤 맛집을 찾으시나요? 상세히 알려주세요!")
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    process_chat(user_input)

def process_chat(user_input):
    # JSON 프롬프트 생성
    json_prompt = f"""질문에서 요구사항을 보고 JSON의 모든 항목(is_recommend, 주소, 업종, 이용건수구간, 이용금액구간, 건당평균이용금액구간)을 반드시 반환하라\n
        각 필드의 대한 설명이다. address:주소(예. 제주시 ㅁㅁ읍),    category:업종,   Usage_Count_Range:이용건수구간(예. 이용건수 상위 N%),  Spending_Amount_Range:이용금액구간(예. 이용금액구간 상위 N%),
        Average_Spending_Amount_Range:건당평균이용금액구간(예. 건당평균이용금액 상위 N%), is_recommend:(추천 혹은 연계되는 질문일경우(True. 예시: 추천해줘), 조건에 따른 검색일경우(False. 예시:조건이 XX하고, 가장 XX한것은?))\n
        ranking_condition는 없을 수도 있으며, 오직 순위를 나타내는 조건(가장 큰것, 가장 작은것)에만 해당한다. 
        \n질문: {user_input}"""
        
    recommend_prompt = f"""당신은 제주도의 맛집을 소개하는 밝고 친근한 가이드입니다! 🌊

📍 첫 번째로 할 일: 사용자 질문을 분석하여 가장 적절한 함수를 선택하고 즉시 실행하세요.

사용 가능한 함수들:
1. search_by_location(region, type, preference)
  언제 사용하나요?
  - 특정 지역의 특정 종류 맛집을 찾을 때
  - 예시 질문: "제주시 칠성로 근처 카페", "서귀포 횟집 추천"
  - 선호조건(뷰, 분위기 등)이 있을 때 preference 활용

2. search_nearby_location(place)
  언제 사용하나요?
  - 특정 장소나 랜드마크 근처 맛집을 찾을 때
  - 예시 질문: "제주공항 근처 맛집", "성산일출봉 주변 카페"

✨ 답변 작성 가이드:
1. 시작하는 말
  - "안녕하세요! [사용자 요청 맛집] 찾으셨네요!"
  - "제가 딱 좋은 곳을 알고 있어요!"
  - "특별한 곳을 소개해드릴게요!"

2. 추천 스토리텔링
  - 마치 친구에게 맛집을 추천하듯 자연스럽게
  - 리뷰와 설명을 자연스럽게 녹여서 설명
  - 장점을 과하지 않게 매력적으로 전달
  예시:
  "이곳은 [특징]으로 유명한데요, 실제로 방문하신 분들도 [리뷰 내용]라고 하시더라고요!"
  "[특별한 점]이 매력적인 곳이에요. [구체적인 설명]"

3. 실용적인 정보 전달
  - 위치, 대표 메뉴, 특징을 자연스럽게 설명
  - 이용 팁이나 추천 포인트 포함
  예시:
  "특히 [시간대]에 방문하시면 [장점]을 제대로 즐기실 수 있어요!"
  "[메뉴나 특징]이 이곳의 차별점인데요,"

4. 마무리
  - 방문 시 도움될 만한 팁 추가
  - 친근하고 긍정적인 마무리
  예시:
  "방문하시면 후회하지 않으실 거예요!"
  "이런 특별한 경험 어떠세요?"

⚠️ 주의사항:
- 검색 결과의 실제 데이터만 사용하기
- 과장된 표현 피하기
- 자연스러운 대화체 사용하기
- 이모지 적절히 활용하기

💡 중요:
- 먼저 함수를 실행하여 실제 데이터 확보
- 데이터 기반으로 스토리텔링
- 친근하고 신뢰감 있는 톤 유지

이제 적절한 함수를 선택하여 실행하고, 결과를 바탕으로 친근하게 답변해주세요!"
현재 사용자 질문: "{user_input}"""
    
    # JSON 응답 처리
    response = model.generate_content(json_prompt, 
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", 
            response_schema=list[Query]
        ),
        tools=[])
    
    dic = merge_dicts(json.loads(response.parts[0].text))
    
    # 추천 vs 검색 분기
    if not dic.get("is_recommend"):
        response2 = model.generate_content(json_prompt, 
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[Query]
            ),
            tools=[])
        dic2 = merge_dicts(json.loads(response2.parts[0].text))
        
        correct_dic = dic if len(dic) > len(dic2) else dic2
        output = filtering(correct_dic)
        st.session_state.history.append({
            'role': 'model',
            'parts': [{'text': output}]
        })
        st.markdown(output, unsafe_allow_html=True)
    else:
        output = st.session_state.chat_session.send_message(
            recommend_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="text/plain",
                response_schema=None,
                temperature=0.1
            ),
            tools=function_repository.values()
        )
        st.markdown(output.text)
        st.session_state.history.append({
            'role': 'model',
            'parts': [{'text': output.text}]
        })

# 전역변수 초기화 및 실행
model = load_model()
embed_model, index, nearby_index, name_index = load_index()
df = load_data()

if __name__ == "__main__":
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False
    if "system_message_displayed" not in st.session_state:
        st.session_state.system_message_displayed = False
    if "options" not in st.session_state:
        st.session_state.options = [None, None, None]
    if "history" not in st.session_state:
        st.session_state.history = []
    if "chat_session" not in st.session_state:    
        st.session_state["chat_session"] = model.start_chat(
            history=[],
            enable_automatic_function_calling=True
        )
    
    main()