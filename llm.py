from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'dive2025home'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_llm(model='gpt-4.1'):
    llm = ChatOpenAI(model=model)
    return llm

def get_dictionary_chain():

    dictionary = [
    # 당사자
    "집주인 -> 임대인",
    "건물주 -> 임대인",
    "세입자 -> 임차인",
    "세든 사람 -> 임차인",
    "전세 사는 사람 -> 임차인",
    "월세 사는 사람 -> 임차인",
    "입주자 -> 임차인",
    "새 집주인 -> 양수인",
    "집 산 사람 -> 양수인",

    # 계약
    "전세계약 -> 임대차",
    "월세계약 -> 임대차",
    "집 계약 -> 임대차",
    "전세 -> 보증금",
    "전세금 -> 보증금",
    "전세 보증금 -> 보증금",
    "월세 -> 차임",
    "집세 -> 차임",
    "보증금 -> 임대차 보증금",
    "집 -> 주택",
    "아파트 -> 주택",
    "빌라 -> 주택",
    "원룸 -> 주택",
    "재계약 -> 계약갱신",
    "계약 연장 -> 계약갱신",
    "자동 연장 -> 묵시적 갱신",
    "계약 자동 연장 -> 묵시적 갱신",
    "계약 만료 -> 임대차기간 만료",
    "집 빌려 쓰기 -> 임대차",
    "집 넘겨받다 -> 주택 인도",
    "전입신고 -> 주민등록",
    "계약 해지 통보 -> 계약해지 통지",
    "전전세 -> 전대",
    "관리비 -> 공과금",
    "근저당 -> 담보물권",

    # 권리·보장
    "세입자 권리 -> 대항력",
    "집주인 바뀌어도 계약 유지 -> 임대인의 지위 승계",
    "날짜 도장 -> 확정일자",
    "보증금 먼저 돌려받는 권리 -> 우선변제권",
    "보증금 먼저 받기 -> 우선변제권",
    "작은 보증금 보호 -> 소액보증금 보호",
    "임차권 등기 -> 임차권등기명령",
    "보증금 못 받았을 때 하는 등기 -> 임차권등기명령",
    "집 등기부에 전세 기록 -> 임대차등기",
    "보증금 반환 요구권 -> 보증금반환청구권",
    "보증금 돌려받기 -> 보증금의 회수",
    "보증금 반환 -> 보증금의 회수",
    "계약 갱신 요구 -> 계약갱신요구권",
    "전입신고와 확정일자 -> 대항요건과 확정일자",

    # 상황/사건
    "집주인이 바뀌었을 때 -> 임차주택의 양도",
    "집이 팔렸을 때 -> 임차주택의 양도",
    "집이 경매로 넘어가다 -> 주택 경매",
    "집이 경매에 넘어갔을 때 -> 경매에 의한 임차권의 소멸",
    "집 압류 -> 경매 진행",
    "임차권 소멸 -> 경매로 인한 임차권 소멸",

    # 분쟁 및 절차
    "보증금 반환 소송 -> 보증금반환청구소송",
    "소액사건 소송 -> 소액사건심판법 준용",
    "분쟁조정위원회 -> 주택임대차분쟁조정위원회",
    "법률구조공단 조정 -> 대한법률구조공단 조정위원회",
    "분쟁조정 -> 조정절차",
    "합의서 -> 조정서",

    # 특수 규정
    "미등기 전세 -> 미등기 전세 (보증금 간주)",
    "주택도시기금 전세 -> 전세임대주택 지원",
    "법인 직원 주거용 전세 -> 법인 임대차",
    "조정 결과 효력 -> 집행권원",

    # 기타
    "집 담보 대출 -> 저당권 설정",
    "은행이 먼저 가져간다 -> 우선변제권 행사",
    "세금 체납 조회 -> 납세증명서",
    "세입자 보호법 -> 주택임대차보호법",
    "강제 규정 -> 강행규정",
    "법 어기면 무효 -> 임차인에게 불리한 약정은 무효"
]
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain

def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    system_prompt = (
        "당신은 주택임대차보호법 전문가입니다. 사용자의 주택임대차보호법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 주택임대차보호법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain

def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    home_chain = {"input": dictionary_chain} | rag_chain
    ai_response = home_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response
