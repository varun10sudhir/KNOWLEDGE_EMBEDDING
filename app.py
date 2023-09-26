import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

loader = CSVLoader(file_path="dataset.csv")
documents = loader.load()


embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents,embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class lawyer and a master of the law. 
I will share a question a person has about the law and you must answer in the best manner possible with high accuracy

Answer with a detialled understanding of the law

Below is a message I received from the prospect:
{message}

Please write the best response that I should send to this prospect:
"""
prompt = PromptTemplate(
    input_variables=["message"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

message = """What are my fundamental rights?"""

print(generate_response(message))