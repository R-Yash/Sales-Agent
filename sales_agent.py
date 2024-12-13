from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key = OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

store = {}

def get_vector_store(PATH):
    loader = CSVLoader(file_path=PATH, encoding="utf-8",csv_args={'delimiter': ','})
    items = loader.load()
    db = Chroma.from_documents(items, embeddings,persist_directory='db/')
    return db.as_retriever()

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

try:
    retriever = Chroma(persist_directory='db/',embedding_function=embeddings).as_retriever()
except:
    retriever = get_vector_store('products.csv')

history_instruction = (
    """
    You are given the history of the conversation up to this point.
    Extract important points from the history, such as the user's preferences and instructions.
    Summarize the history and use it to provide context for the current conversation.
    """
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", history_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
history_added = create_history_aware_retriever(
    llm, retriever, prompt
)

system_prompt = (
    """
    You are an Intelligent Bot that can talk to the user and help him with his requests. 
    You will use the previous messages with the user in the form of memory to help him. You also have a list of relevent product data as a context to help the user. 
    If there are no previous messages, you can start the conversation by asking the user what he needs help with.
    If data is insufficient, you can ask the user for more information.
    \n\n
    context: {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(history_added, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

print("Starting Bot")

while True:
    inp = input("User: ")
    if inp == "exit":
        break
    
    
    ans = conversational_rag_chain.invoke(
        {"input": inp},
        config={"configurable": {"session_id": "abc123"}}
    )["answer"]

    print(ans)
    print('\n')