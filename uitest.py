import streamlit as st

# 세션 상태 초기화
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False  # 폼 제출 상태 확인
if "system_message_displayed" not in st.session_state:
    st.session_state.system_message_displayed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 채팅 내역 저장
if "options" not in st.session_state:
    st.session_state.options = [None, None, None]

st.set_page_config(page_title="🍊제주 맛집 추천")

st.title("반갑다!👋")
st.subheader("제주 맛집을 추천해주겠다")
st.write("")

# 옵션 선택 폼
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

        # submit_button = st.form_submit_button(label="완료")
        # cancel_button = st.form_submit_button(label="생략")
    
    if submit_button:
        st.session_state.form_submitted = True
        st.session_state.options[:] = [option1, option2, option3]
        # 선택한 옵션을 채팅 기록에 추가
        st.session_state.chat_history.append({"role": "assistant", "content": f"선택된 나이대: {option1}, 방문 시간: {option2}, 현지인 기준: {option3}"})
        st.rerun()
        
    if cancel_button:
        st.session_state.form_submitted = True
        # 선택한 옵션을 채팅 기록에 추가
        st.session_state.chat_history.append({"role": "assistant", "content": f"옵션 생략"})
        st.rerun()

user_input = None

# 시스템 메시지: 처음 한 번만 표시
if not st.session_state.system_message_displayed and st.session_state.form_submitted:
    system_message = "반갑다! 무엇을 도와줄까? 질문을 입력하라."
    st.session_state.chat_history.append({"role": "assistant", "content": system_message})
    st.session_state.system_message_displayed = True

if st.session_state.form_submitted:
    # 채팅 입력 받기
    user_input = st.chat_input("메시지를 입력하세요")   

# 채팅 기록 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])





# 사용자 입력 처리
if user_input:
    # 사용자 메시지를 기록
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # LLM 응답 예시 (현재는 비어있음)
    llm_response = f"LLM의 응답 예시: {user_input}에 대한 답변입니다."

    # LLM 응답을 대화 기록에 추가
    st.session_state.chat_history.append({"role": "assistant", "content": llm_response})

    # 대화 기록 갱신
    st.rerun()
