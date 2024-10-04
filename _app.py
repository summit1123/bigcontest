import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss
import time as ttime

import streamlit as st

# 경로 설정
data_path = './data'
module_path = './modules'

# Gemini 설정
import google.generativeai as genai

# import shutil
# os.makedirs("/root/.streamlit", exist_ok=True)
# shutil.copy("secrets.toml", "/root/.streamlit/secrets.toml")

# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# 필터링 기준 - 나중엔 좀 조절하긴해야함. 다 푸는건 별로겠지?
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

# Gemini 모델 선택
model = genai.GenerativeModel("gemini-1.5-flash", safe, )



# CSV 파일 로드
## 자체 전처리를 거친 데이터 파일 활용
csv_file_path = "JEJU_MCT_DATA_modified.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path))

# 최신연월 데이터만 가져옴
df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)


# Streamlit App UI

st.set_page_config(page_title="🍊참신한 제주 맛집!")

# Replicate Credentials
with st.sidebar:
    st.title("🍊참신한! 제주 맛집")

    st.write("")

    st.subheader("언드레 가신디가?")

    # selectbox 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stSelectbox label {  /* This targets the label element for selectbox */
            display: none;  /* Hides the label element */
        }
        .stSelectbox div[role='combobox'] {
            margin-top: -20px; /* Adjusts the margin if needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    time = st.sidebar.selectbox("", ["아침", "점심", "오후", "저녁", "밤"], key="time")

    st.write("")

    st.subheader("어드레가 맘에 드신디가?")

    # radio 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stRadio > label {
            display: none;
        }
        .stRadio > div {
            margin-top: -20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    local_choice = st.radio(
        '',
        ('제주도민 맛집', '관광객 맛집')
    )

    st.write("")

st.title("혼저 옵서예!👋")
st.subheader("군맛난 제주 밥집🧑‍🍳 추천해드릴게예")

st.write("")

st.write("#흑돼지 #갈치조림 #옥돔구이 #고사리해장국 #전복뚝배기 #한치물회 #빙떡 #오메기떡..🤤")

st.write("")

image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTHBMuNn2EZw3PzOHnLjDg_psyp-egZXcclWbiASta57PBiKwzpW5itBNms9VFU8UwEMQ&usqp=CAU"
image_html = f"""
<div style="display: flex; justify-content: center;">
    <img src="{image_path}" alt="centered image" width="50%">
</div>
"""
st.markdown(image_html, unsafe_allow_html=True)

st.write("")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# RAG

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face의 사전 학습된 임베딩 모델과 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

print(f'Device is {device}.')


# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    """a
    FAISS 인덱스를 파일에서 로드합니다.

    Parameters:
    index_path (str): 인덱스 파일 경로.

    Returns:
    faiss.Index: 로드된 FAISS 인덱스 객체.
    """
    if os.path.exists(index_path):
        # 인덱스 파일에서 로드
        index = faiss.read_index(index_path)
        print(f"FAISS 인덱스가 {index_path}에서 로드되었습니다.")
        return index
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")

# 텍스트 임베딩
def embed_text(text):
    # 토크나이저의 출력도 GPU로 이동
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        # 모델의 출력을 GPU에서 연산하고, 필요한 부분을 가져옴
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()  # 결과를 CPU로 이동하고 numpy 배열로 변환

# 임베딩 로드
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))

def generate_response_with_faiss(question, df, embeddings, model, embed_text, time, local_choice, index_path=os.path.join(module_path, 'faiss_index.index'), max_count=10, k=5, print_prompt=True):
    filtered_df = df
    
    start_time = ttime.time()
    # FAISS 인덱스를 파일에서 로드
    index = load_faiss_index(index_path)

    # 검색 쿼리 임베딩 생성
    query_embedding = embed_text(question).reshape(1, -1)

    # 가장 유사한 텍스트 검색 (3배수)
    distances, indices = index.search(query_embedding, k*3)

    # FAISS로 검색된 상위 k개의 데이터프레임 추출
    filtered_df = filtered_df.iloc[indices[0, :]].copy().reset_index(drop=True)
    end_time = ttime.time()
    latency = end_time - start_time
    print(f"Index Latency: {latency:.6f} seconds")

    # 웹페이지의 사이드바에서 선택하는 영업시간, 현지인 맛집 조건 구현

    # 영업시간 옵션
    # 필터링 조건으로 활용

    # 영업시간 조건을 만족하는 가게들만 필터링
    if time == '아침':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(5, 12)))].reset_index(drop=True)
    elif time == '점심':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(12, 14)))].reset_index(drop=True)
    elif time == '오후':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(14, 18)))].reset_index(drop=True)
    elif time == '저녁':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(18, 23)))].reset_index(drop=True)
    elif time == '밤':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in [23, 24, 1, 2, 3, 4]))].reset_index(drop=True)

    # 필터링 후 가게가 없으면 메시지를 반환
    if filtered_df.empty:
        return f"현재 선택하신 시간대({time})에는 영업하는 가게가 없습니다."

    filtered_df = filtered_df.reset_index(drop=True).head(k)


    # 현지인 맛집 옵션

    # 프롬프트에 반영하여 활용
    if local_choice == '제주도민 맛집':
        local_choice = '제주도민(현지인) 맛집'
    elif local_choice == '관광객 맛집':
        local_choice = '현지인 비중이 낮은 관광객 맛집'

    # 선택된 결과가 없으면 처리
    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."


    # 참고할 정보와 프롬프트 구성
    reference_info = ""
    for idx, row in filtered_df.iterrows():
        reference_info += f"{row['text']}\n"

    # 응답을 받아오기 위한 프롬프트 생성
    # prompt = f"질문: {question}\n조건: 이용 비중은 소수가 아닌 %로 나타내. 위치는 제대로 확인하고 결정해\n참고할 정보:\n{reference_info}\n응답:"
    # prompt = f"내가 질문할 문장에서 찾고싶은 지명, 먹고싶은 음식종류, 현재 위치(현재위치는 가능하다면 문장을 통해 추론해)등의 정보를 찾아서 다음과 같은 형식으로 출력해 없다면 0을 넣어. 형식: LOC:찾고싶은 지명 FOOD:음식종류 NOW:현재위치 INFO:요청사항\n질문: {question}"
    prompt = f"너는 맛집을 추천해주는 사람인데, 너에게 음식에 관한 질문을 할꺼야. 뒤에 질문할 문장에서 긍정적인 부분(원하는 부분. 없으면 좋겠다는 넣지마) 부정적인 부분(싫어하는, 제외할 부분)을 분리해서 다음과 같은 형식으로 출력해줘. 참고로 흑돼지는 제주도의 특산품인 돼지고기인데 이 내용을 뒤에 출력하지 말아줘. 질문에서 없는 내용을 적지마\n형식: POS:긍정 NEG:부정\n질문: {question}"
    if print_prompt:
        print('-----------------------------'*3)
        print(prompt)
        print('-----------------------------'*3)



    start_time = ttime.time()
    # 응답 생성
    response = model.generate_content(prompt)
    end_time = ttime.time()
    latency = end_time - start_time
    print(f"LLM Latency: {latency:.6f} seconds")
    
    return response


# User-provided prompt
if prompt := st.chat_input(): # (disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # response = generate_llama2_response(prompt)
            response = generate_response_with_faiss(prompt, df, embeddings, model, embed_text, time, local_choice)
            placeholder = st.empty()
            full_response = ''

            # 만약 response가 GenerateContentResponse 객체라면, 문자열로 변환하여 사용합니다.
            if isinstance(response, str):
                full_response = response
            else:
                full_response = response.text  # response 객체에서 텍스트 부분 추출

            # for item in response:
            #     full_response += item
            #     placeholder.markdown(full_response)

            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)