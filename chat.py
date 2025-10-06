import streamlit as st
from dotenv import load_dotenv
import llm

# env 설정
load_dotenv()

st.set_page_config( page_title = 'chatbot', page_icon="😊" )
st.title("😊 챗봇")
st.caption("원자력안전법에 대한 질문-답변")


# 대화이력 초기화
if 'message_list' not in st.session_state :
    st.session_state.message_list = []

# 대화이력(사용자, 답변) 만 페이지에 로딩
for message in st.session_state.message_list :
    with st.chat_message(message['role']):
        st.write(message['content'])
        
# 대화내용을 입력받아서 처리
if user_question := st.chat_input(placeholder="원자력안전법 관련 질문 넣기") :
    with st.chat_message("user") :
        st.write(user_question)
    st.session_state.message_list.append({"role":"user", "content": user_question})
    
    with st.spinner("답변을 생성중 입니다.") :
        ai_response = llm.get_ai_message(user_question)
        with st.chat_message("ai") :
            # st.write(ai_message) # get_message일떄 사용.
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role":"ai", "content": ai_message}) # stream 최종에서 묶인 데이터를 인메모리 저장
        # st.session_state.message_list.append({"role":"ai", "content": ai_response})
print(f'after === {st.session_state.message_list}')
