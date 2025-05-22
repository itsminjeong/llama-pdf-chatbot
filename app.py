import os
import tempfile
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

st.set_page_config(page_title="LLaMA PDF Chatbot", page_icon="🦙")

# ✅ 환경 변수 로드
load_dotenv()

# ✅ 언어 선택
lang = st.sidebar.selectbox("Language / 언어 선택", ("한국어", "English"))

# ✅ 다국어 메시지 정의
messages = {
    "한국어": {
        "page_title": "LLaMA 기반 PDF 챗봇",
        "page_description": "PDF를 업로드하고 자유롭게 질문하세요.",
        "file_uploader": "📄 PDF 업로드",
        "input_label": "질문을 입력하세요",
        "input_placeholder": "예: 2페이지 요약해줘",
        "send_button": "보내기",
        "init_greeting_user": "안녕하세요!",
        "init_greeting_bot": "안녕하세요! 문서에 대해 무엇이든 물어보세요."
    },
    "English": {
        "page_title": "LLaMA-based PDF Chatbot",
        "page_description": "Upload a PDF and ask your questions freely.",
        "file_uploader": "📄 Upload PDF",
        "input_label": "Enter your question",
        "input_placeholder": "e.g., Summarize page 2",
        "send_button": "Send",
        "init_greeting_user": "Hello!",
        "init_greeting_bot": "Hello! Ask me anything about the uploaded document."
    }
}

msg = messages[lang]

# ✅ 페이지 UI 출력
st.title(f"🦙 {msg['page_title']}")
st.markdown(msg["page_description"])

# ✅ PDF 업로드
uploaded_file = st.sidebar.file_uploader(msg["file_uploader"], type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # ✅ 문서 로드
    loader = PyPDFLoader(tmp_path)
    data = loader.load()

    # ✅ 임베딩 생성 (CPU 강제 설정)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={"device": "cpu"}
    )
    vectordb = FAISS.from_documents(data, embeddings)

    # ✅ LLaMA 모델 로딩
    llm = LlamaCpp(
        model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        temperature=0.0,
        max_tokens=1024,
        n_ctx=2048,
        verbose=True,
    )

    # ✅ RAG 체인 구성
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    # ✅ 세션 초기화
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = [msg["init_greeting_bot"]]
    if "past" not in st.session_state:
        st.session_state["past"] = [msg["init_greeting_user"]]

    # ✅ 질문 입력 폼
    with st.container():
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(msg["input_label"], placeholder=msg["input_placeholder"])
            submitted = st.form_submit_button(label=msg["send_button"])

            if submitted and user_input:
                # ✅ 프롬프트 단순화: 사용자 질문만 전달
                result = chain({
                    "question": user_input,
                    "chat_history": st.session_state["history"]
                })
                answer = result["answer"]

                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(answer)
                st.session_state["history"].append((user_input, answer))

    # ✅ 대화 출력
    if st.session_state["generated"]:
        with st.container():
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=f"user_{i}", avatar_style="fun-emoji")
                message(st.session_state["generated"][i], key=f"bot_{i}", avatar_style="bottts")
