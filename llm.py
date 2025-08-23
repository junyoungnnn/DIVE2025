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
from langchain_core.runnables import RunnablePassthrough

from config import legal_examples, advice_examples, term_examples

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_llm(model='gpt-4.1'):
    llm = ChatOpenAI(model=model)
    return llm

def get_retriever_home():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'dive2025home'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_retriever_rentalscam():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'homeandrentalscam'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_history_aware_retriever(llm, retriever):
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

# 1단계: 질문을 법률 용어로 변환
def get_dictionary_chain():

    llm = get_llm()

    dictionary = [

    "집주인 -> 임대인",
    "건물주 -> 임대인",
    "세입자 -> 임차인",
]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 아래 사전을 참고해서 질문에 사용된 일상 용어를 법률 용어로 변경해주세요.
        질문의 의미가 바뀌지 않는 선에서 자연스럽게 수정하고, 수정된 질문만 간결하게 반환해주세요.
        만약 수정할 필요가 없다고 판단되면, 원래 질문을 그대로 반환해주세요.

        사전: {dictionary}
        
        질문: {{question}}
    """)
    return prompt | llm | StrOutputParser()

# 2단계: 질문 의도 분류기
def get_intent_classifier_chain():
    llm = get_llm()
    
    intent_classifier_prompt_text = """
    당신은 사용자의 질문 의도를 4가지 카테고리로 분류하는 AI 어시스턴트입니다.
    사용자의 질문을 읽고, 아래 카테고리 중 가장 적합한 하나만 골라 그 이름만 답변해주세요.
    다른 설명은 절대 추가하지 마세요.

    ---
    [카테고리 설명]
    1. legal_question (법률 질문): 법률 용어, 조항, 절차 등 법 자체에 대한 질문.
    2. actionable_advice_question (행동 요령 질문): '어떻게 해야 하는지' 구체적인 행동 방법을 묻는 질문.
    3. term_question (용어 설명 질문): '임대인', '확정일자' 등 특정 용어의 뜻을 묻는 질문.
    4. irrelevant_question (관련 없는 질문): 위 세 가지에 속하지 않는 모든 질문.
    ---

    [사용자 질문]
    {question}

    [분류]
    """
    prompt = ChatPromptTemplate.from_template(intent_classifier_prompt_text)
    return prompt | llm | StrOutputParser()

# 3단계: 역할별 전문가 체인 (Specialists)

def get_base_rag_chain(llm, retriever, system_prompt_template, fewshot_example):
    # FewShot을 포함한 범용 RAG 체인을 생성하는 함수
    history_aware_retriever = get_history_aware_retriever(llm, retriever)
    
    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{answer}")])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=fewshot_example,
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_legal_expert_chain():
    llm = get_llm()
    retriever = get_retriever_home()
    system_prompt = (
        "당신은 주택임대차보호법 전문가예요. 사용자의 주택임대차보호법에 관한 질문에 답변해주세요.\n"
        "아래에 제공된 문서를 활용해서 답변해주시고, 답변을 알 수 없다면 모른다고 답변해주세요.\n"
        "답변을 제공할 때는 '주택임대차보호법(제XX조)에 따르면,' 이라고 시작하면서 답변해주시고, 사회초년생이 이해하기 쉽게 예시를 들어 설명해주세요.\n"
        "답변은 2-5 문장 정도로 간결하고 명확하게 작성해주세요.\n\n"
        "{context}"
    )
    return get_base_rag_chain(llm, retriever, system_prompt, legal_examples)

def get_action_advice_expert_chain():
    llm = get_llm()
    retriever = get_retriever_rentalscam()
    system_prompt = (
        "당신은 전세 계약 경험이 많은 선배예요. 아래에 제공된 법률 문서를 참고하여, 사용자가 처한 상황에서 어떤 행동을 해야 할지 질문에 맞춰 단계별로 알려주세요.\n"
        "사용자가 따라 하기 쉽도록, 답변을 번호나 글머리 기호를 사용한 '체크리스트' 또는 '단계별 절차' 형식으로 제공해주세요.\n"
        "아래에 제공된 문서와 관련된 법 조항이 있다면 '주택임대차보호법 및 전세사기 피해지원법(제XX조)에 따르면,' 이라고 시작하면서 답변해주시고, 딱딱한 법률 용어보다는 부드럽고 실용적인 조언 형태로 작성해주세요.\n"
        "답변은 2-5 문장 정도로 간결하고 명확하게 작성해주세요.\n\n"
        "{context}"
    )
    return get_base_rag_chain(llm, retriever, system_prompt, advice_examples)

def get_term_expert_chain():
    llm = get_llm()
    retriever = get_retriever_home()
    system_prompt = (
        "당신은 법률 용어를 아주 쉽게 설명해주는 친절한 선배예요. 사회초년생의 눈높이에 맞춰, 어려운 법률 용어를 일상적인 예시나 비유를 들어 설명해주세요.\n"
        "아래에 제공된 참고 자료를 활용하되, 딱딱한 법률 조항을 나열하기보다는 핵심 의미를 풀어서 전달하는 데 집중해주세요.\n"
        "답변은 2-5 문장 정도로 간결하고 명확하게 작성해주세요.\n\n"
        "{context}"
    )
    return get_base_rag_chain(llm, retriever, system_prompt, term_examples)

# 4단계: 메인 로직

def get_ai_response(user_message):
    # 1. 질문을 법률 용어로 변환
    dictionary_chain = get_dictionary_chain()
    normalized_question = dictionary_chain.invoke({"question": user_message})
    
    # 2. 질문 의도 분류
    intent_classifier = get_intent_classifier_chain()
    intent = intent_classifier.invoke({"question": normalized_question}) # 분류는 변환된 질문으로

    # 3. 의도에 따라 다른 체인 실행
    if intent == "legal_question":
        chain = get_legal_expert_chain()
    elif intent == "actionable_advice_question":
        chain = get_action_advice_expert_chain()
    elif intent == "term_question":
        chain = get_term_expert_chain()
    else: # irrelevant_question
        return iter(["죄송합니다, 저는 주택 임대차 계약과 관련된 법률 및 위험도 분석에 대해서만 답변을 드릴 수 있어요."])

    # RAG 체인에 메모리 기능 추가
    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    # 스트리밍 답변 반환
    return conversational_rag_chain.stream(
        {"input": normalized_question},
        config={"configurable": {"session_id": 'abc123'}},
    )