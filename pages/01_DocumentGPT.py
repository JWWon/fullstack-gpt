from operator import itemgetter
import os
from typing import Any, List, Literal
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Document GPT", page_icon="📜")

st.title("Document GPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions about the document.

Listed items below are required to use the chatbot:

- Upload a .txt .pdf or .docx file
- Fill in your OpenAI API key
"""
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that answers questions using the following context and chat history.
            If you don't know the answer, just say that you don't know.

            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def save_message(message: str, role: Literal["user", "assistant", "ai", "human"]):
    st.session_state.messages.append({"role": role, "message": message})


def send_message(
    message: str, role: Literal["user", "assistant", "ai", "human"], save=True
):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def print_history():
    for m in st.session_state.messages:
        send_message(m["message"], m["role"], save=False)


def load_memory():
    for i in range(len(st.session_state.messages) // 2):
        human_message = st.session_state.messages[i * 2]["message"]
        ai_message = st.session_state.messages[i * 2 + 1]["message"]
        memory.save_context({"input": human_message}, {"output": ai_message})


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


class ChatCallbackHandler(BaseCallbackHandler):
    message: str = ""
    message_box = st.empty()

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        super().on_llm_start(*args, **kwargs)
        self.message_box = st.empty()

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        super().on_llm_end(*args, **kwargs)
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args: Any, **kwargs: Any) -> None:
        super().on_llm_new_token(token, *args, **kwargs)
        self.message += token
        self.message_box.markdown(self.message)


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        st.secrets.OPENAI_API_KEY if "OPENAI_API_KEY" in st.secrets else "",
        type="password",
    )

    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", type=["txt", "pdf", "docx"]
    )

if openai_api_key:
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    llm = ChatOpenAI(
        api_key=openai_api_key,
        temperature=0,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )

    @st.cache_resource(show_spinner="Embedding file...")
    def embed_file(up: UploadedFile):
        file_content = up.read()
        file_path = f"./.cache/files/{up.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n", chunk_size=600, chunk_overlap=100
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)

        embedding_path = f"./.cache/embeddings/{up.name}"
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        cache = LocalFileStore(embedding_path)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache)

        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        return vectorstore.as_retriever()

    if file:
        retriever = embed_file(file)

        send_message("I'm ready to answer your questions!", "ai", save=False)
        print_history()
        load_memory()

        message = st.chat_input("Ask a question")
        if message:
            send_message(message, "human")

            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnablePassthrough.assign(
                    history=RunnableLambda(memory.load_memory_variables)
                    | itemgetter("history")
                )
                | prompt
                | llm
            )

            with st.chat_message("ai"):
                chain.invoke(message)
    else:
        st.session_state.messages = []
