from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import answer_examples

def get_llm(model = "gpt-4o-mini") :
    llm = ChatOpenAI(model = model) # llm 연계
    return llm

def get_dictionary_chain() :
    dictionary = ["원자력안전법 -> 방사선안전법"]
    rephrase_prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다. 이 경우엔, 답변만 반환합니다.
        사전: {dictionary}
        질문: {{question}}
    """)
    # print(f'1st rephrase_prompt: {rephrase_prompt}')

    llm = get_llm()
    # 발화재구성 - 모델을 통해 답변을 받아 전처리 해주는 코드.
    dictionary_chain = rephrase_prompt | llm | StrOutputParser()
    return dictionary_chain

def get_retriever():
    embedding = OpenAIEmbeddings(model = 'text-embedding-3-large'
                                , dimensions=1024) # 벡터 dimension에 맞게 조절가능. default openAI embedding: 3072
    index_name = 'law-index' # 벡터색인명
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding) # Pinecone 연결
    retriever = database.as_retriever() # 유사도검색
    return retriever

def get_qa_chain() :
    prompt = hub.pull('rlm/rag-prompt') # RAG 기본 프롬프트
    
    llm = get_llm()
    retriever = get_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs = {"prompt": prompt}
    )
    return qa_chain



store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_history_retriever () :
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
        "원자력안전법에 대해서 알려주세요"
        "원자력안전법은~ 으로 시작하고"
        "각 답변은 2-3 문장정도의 짧은 내용의 답변을 원합니다."
        "답변은 개괄식으로, 강조되는 bold체, 이모티콘까지 활용하여 가독성이 좋게 만듭니다."
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
    history_retriever = get_history_retriever ()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input", # {input}
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer') # answer만 답변하기.
    
    return conversational_rag_chain

def get_ai_message(user_question) :
    # 재구성된 답변으로 최종 LLM에 질의처리.
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    law_chain = {"input": dictionary_chain} | rag_chain  # qa_query: query -> rag_chain: input
    # ai_response = law_chain.invoke( # 메세지 답변 (완성본)
    #     {
    #         "question": user_question
    #     }, 
    #     config = {
    #         "configurable":{"session_id": "abc123"}
    #         }
    #     ) # session_id -> get_session_history
    
    ai_response = law_chain.stream( # data:{~}
        {
            "question": user_question
        }, 
        config = {
            "configurable":{"session_id": "abc123"}
            }
        ) # session_id -> get_session_history

    print(f'ai_response: {ai_response}')
    return ai_response

