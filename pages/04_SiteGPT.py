import os
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import streamlit as st
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate

st.set_page_config(page_title="Site GPT", page_icon="🌐")

st.title("Site GPT")

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                
    Examples:
                                                
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                
    Your turn!
    Question: {question}
    """
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    if header:
        header.decompose()

    footer = soup.find("footer")
    if footer:
        footer.decompose()

    return re.sub(r"\W+", " ", str(soup.get_text()).replace("\n", " "))


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        st.secrets.OPENAI_API_KEY if "OPENAI_API_KEY" in st.secrets else "",
        type="password",
    )

    url = st.text_input(
        "Enter a stiemap URL",
        placeholder="https://example.com/sitemap.xml",
        value="https://developers.cloudflare.com/sitemap-0.xml",
    )

is_url_valid = url and ".xml" in url
is_openai_key_valid = openai_api_key


if is_url_valid and is_openai_key_valid:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    @st.cache_resource(show_spinner="Loading website...")
    def indexing_web_contents(_url: str):
        loader = SitemapLoader(
            _url,
            parsing_function=parse_page,
            filter_urls=[r"^.*\/(ai-gateway|vectorize|workers-ai)\/.*"],
        )
        loader.requests_per_second = 100
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = loader.load_and_split(text_splitter=splitter)
        st.write(docs[:5])

        domain = urlparse(_url).netloc
        embedding_path = f"./.cache/site_embeddings/{domain}"
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        store = LocalFileStore(embedding_path)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, store)

        vectorsore = FAISS.from_documents(docs, cached_embeddings)
        return vectorsore.as_retriever()

    def get_answers(inputs):
        docs = inputs["docs"]
        query = inputs["question"]

        answers_chain = answers_prompt | llm

        return {
            "question": query,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {"context": doc.page_content, "question": query}
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in docs
            ],
        }

    def choose_best_answer(inputs):
        query = inputs["question"]
        answers = inputs["answers"]

        choose_chain = choose_prompt | llm

        condenced = "\n\n".join(
            f"Answer:{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}"
            for answer in answers
        )
        return choose_chain.invoke({"answers": condenced, "question": query})

    retriever = indexing_web_contents(url)

    chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_best_answer)
    )

    question = st.text_input("Ask a question")
    if question:
        result = chain.invoke(question)
        st.write(result.content.replace("$", "\$"))
else:
    st.markdown(
        """
        Ask questions about the content of the website.
        """
    )

    if not is_url_valid:
        st.sidebar.error("Please enter a valid sitemap URL")
    if not is_openai_key_valid:
        st.sidebar.error("Please enter an OpenAI API key")
