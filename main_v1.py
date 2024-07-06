# Project LLM_summacum #
# Author: Yohan Park #

# standard imports
import os
import warnings
from pprint import pprint
import time

# 3rd party imports
from dotenv import load_dotenv
from openai import OpenAI
import oracledb

# upstage imports
from langchain_upstage import (
     UpstageLayoutAnalysisLoader,
     UpstageEmbeddings,
     ChatUpstage
)
from langchain_core.messages import HumanMessage, SystemMessage

# langchain imports
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore")

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.docs_dict = {}
    
    def load_data(self):
        counter = 1
        for filename in os.listdir(self.data_path):
            if filename.endswith(".pdf"):
                new_filename = f"{counter}.pdf"
                old_file_path = os.path.join(self.data_path, filename)
                new_file_path = os.path.join(self.data_path, new_filename)
                os.rename(old_file_path, new_file_path)
                layzer = UpstageLayoutAnalysisLoader(new_file_path, split="page")
                docs = layzer.load()
                self.docs_dict[new_filename] = docs
                counter += 1
        return self.docs_dict

class UpstageAPI:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["UPSTAGE_API_KEY"], base_url="https://api.upstage.ai/v1/solar"
        )
    
    def chat(self, message):
        chat_result = self.client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
        )
        return chat_result

class OracleDB:
    def __init__(self):
        self.username = os.environ["DB_USER"]
        self.password = os.environ["DB_PASSWORD"]
        self.dsn = os.environ["DSN"]
        self.conn = None
    
    def connect(self):
        try: 
            self.conn = oracledb.connect(user=self.username, password=self.password, dsn=self.dsn)
            print("Connection successful!", self.conn.version)
        except Exception as e:
            print("Connection failed!")
        return self.conn

class EmbeddingSettings:
    def __init__(self, conn, docs):
        self.conn = conn
        self.docs = docs
        self.upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.knowledge_base = OracleVS.from_documents(docs, self.upstage_embeddings, client=conn, 
                                                      table_name="text_embeddings2", 
                                                      distance_strategy=DistanceStrategy.DOT_PRODUCT)
    
    def get_vector_store(self):
        vector_store = OracleVS(client=self.conn, 
                                embedding_function=self.upstage_embeddings, 
                                table_name="text_embeddings2", 
                                distance_strategy=DistanceStrategy.DOT_PRODUCT)
        return vector_store

class LLMInvoker:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatUpstage()
    
    def invokellm(self, question, template):
        prompt = PromptTemplate.from_template(template)
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        response = chain.invoke(question)
        return response

class OracleDBIndex:
    @staticmethod
    def create_index(conn, vector_store):
        oraclevs.create_index(
            client=conn,
            vector_store=vector_store,
            params={
                "idx_name": "ivf_idx1",
                "idx_type": "IVF",
            },
        )

def main():
    data_loader = DataLoader("data/")
    docs_dict = data_loader.load_data()

    upstage_api = UpstageAPI()
    chat_result = upstage_api.chat("What about Korea?")
    pprint(chat_result)

    oracle_db = OracleDB()
    conn = oracle_db.connect()

    embedding_settings = EmbeddingSettings(conn, docs_dict)
    vector_store = embedding_settings.get_vector_store()

    retriever = vector_store.as_retriever()
    llm_invoker = LLMInvoker(retriever)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = llm_invoker.invokellm(user_input, """Answer the question based only on the following context:
                                                        {context} 
                                                        Question: {question} 
                                                     """)
        print("Bot:", response)

    OracleDBIndex.create_index(conn, vector_store)

if __name__ == "__main__":
    main()