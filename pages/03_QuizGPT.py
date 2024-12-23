import os
from typing import List, Union
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_openai.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document, BaseOutputParser

st.set_page_config(page_title="Quiz GPT", page_icon="🎓")

st.title("Quiz GPT")


class JSONOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        import json

        text = text.replace("json", "").replace("```", "")
        return json.loads(text)


output_parser = JSONOutputParser()


@st.cache_data(show_spinner="Loading file...")
def split_file(up: UploadedFile):
    file_content = up.read()
    file_path = f"./.cache/quiz_files/{up.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    return loader.load_and_split(text_splitter=splitter)


@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wikipedia(query: str) -> List[Document]:
    retriever = WikipediaRetriever(top_k_results=3)
    return retriever.get_relevant_documents(query)


def format_docs(d: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in d])


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a helpful assistant that is role playing as a teacher.
                    
                Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
                
                Each question should have 4 answers, three of them must be incorrect and one should be correct.
                    
                Use (o) to signal the correct answer.
                    
                Question examples:
                    
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)
                    
                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998
                    
                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model
                    
                Your turn!
                    
                Context: {context}
            """,
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a powerful formatting algorithm.
        
        You format exam questions into JSON format.
        Answers with (o) are the correct ones.
        
        Example Input:

        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
        
        
        Example Output:
        
        ```json
        {{ "questions": [
                {{
                    "question": "What is the color of the ocean?",
                    "answers": [
                            {{
                                "answer": "Red",
                                "correct": false
                            }},
                            {{
                                "answer": "Yellow",
                                "correct": false
                            }},
                            {{
                                "answer": "Green",
                                "correct": false
                            }},
                            {{
                                "answer": "Blue",
                                "correct": true
                            }},
                    ]
                }},
                            {{
                    "question": "What is the capital or Georgia?",
                    "answers": [
                            {{
                                "answer": "Baku",
                                "correct": false
                            }},
                            {{
                                "answer": "Tbilisi",
                                "correct": true
                            }},
                            {{
                                "answer": "Manila",
                                "correct": false
                            }},
                            {{
                                "answer": "Beirut",
                                "correct": false
                            }},
                    ]
                }},
                            {{
                    "question": "When was Avatar released?",
                    "answers": [
                            {{
                                "answer": "2007",
                                "correct": false
                            }},
                            {{
                                "answer": "2001",
                                "correct": false
                            }},
                            {{
                                "answer": "2009",
                                "correct": true
                            }},
                            {{
                                "answer": "1998",
                                "correct": false
                            }},
                    ]
                }},
                {{
                    "question": "Who was Julius Caesar?",
                    "answers": [
                            {{
                                "answer": "A Roman Emperor",
                                "correct": true
                            }},
                            {{
                                "answer": "Painter",
                                "correct": false
                            }},
                            {{
                                "answer": "Actor",
                                "correct": false
                            }},
                            {{
                                "answer": "Model",
                                "correct": false
                            }},
                    ]
                }}
            ]
        }}
        ```
        Your turn!

        Questions: {context}
     """,
        )
    ]
)


@st.cache_data(show_spinner="Generating quiz...")
def invoke_quiz_chain(
    _key: str, _docs: List[Document], _llm: ChatOpenAI
) -> dict[str, List[dict[str, Union[str, List[dict]]]]]:
    questions_chain = {"context": format_docs} | questions_prompt | _llm
    formatting_chain = (
        {"context": RunnablePassthrough()} | formatting_prompt | _llm | output_parser
    )

    chain = questions_chain | formatting_chain
    return chain.invoke(_docs)


key: str = ""
docs: List[Document] = []
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        st.secrets.OPENAI_API_KEY if "OPENAI_API_KEY" in st.secrets else "",
        type="password",
    )

    choice = st.selectbox("Choose what you want to use.", ("File", "Wikipedia article"))

    if choice == "File":
        file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
        if file:
            key = file.name
            docs = split_file(file)
    elif choice == "Wikipedia article":
        topic = st.text_input("Enter the topic")
        if topic:
            key = topic
            docs = search_wikipedia(topic)

if not docs or not openai_api_key:
    st.markdown(
        """
    Welcome to Quiz GPT!

    This app will help you create a quiz from a document or a Wikipedia article.

    Please upload a file or enter a topic to get started.

    Also, please fill in your OpenAI API key.
    """
    )
else:
    if openai_api_key:
        llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        response = invoke_quiz_chain(key, docs, llm)
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option",
                    question["answers"],
                    format_func=lambda x: x["answer"],
                    index=None,
                )
                if value is not None:
                    if value["correct"] is True:
                        st.success("Correct!")
                    else:
                        st.error("Wrong :(")

            submit = st.form_submit_button("Submit")
