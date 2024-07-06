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
from tavily import TavilyClient as tavily

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
from langchain_core.tools import tool

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.docs_dict_page = {}
        self.docs_dict_chunk = {}
    
    def load_data(self):
        counter = 1
        for filename in os.listdir(self.data_path):
            print("Loading {}-th data...".format(counter))
            if filename.endswith(".pdf"):
                new_filename = f"{counter}.pdf"
                old_file_path = os.path.join(self.data_path, filename)
                new_file_path = os.path.join(self.data_path, new_filename)
                os.rename(old_file_path, new_file_path)
                layzer = UpstageLayoutAnalysisLoader(new_file_path, split="page")
                docs = layzer.load()
                self.docs_dict_page[new_filename] = docs
                counter += 1

        print("Data loaded successfully!")
        return
    
    def chunking(self):
        if self.docs_dict_page == {}:
            print("Have you called load_data()?")
            return False
        
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=1500, chunk_overlap=200, language=Language.HTML
        )

        print("Splitting each document into chunks...")
        for i, docs in self.docs_dict_page.items():
            print("Splitting {}-th file".format(i))
            chunks = text_splitter.split_documents(docs)
            self.docs_dict_chunk[i] = chunks
        
        print("Document splitting completed!")
        return True

# Deprecated
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

class OracleDB:
    def __init__(self, docs_dict):
        self.username = os.environ["DB_USER"]
        self.password = os.environ["DB_PASSWORD"]
        self.dsn = os.environ["DSN"]

        self.docs_dict = docs_dict
        self.upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.conn = None
        self.knowledge_base = None

    def connect(self):
        try: 
            self.conn = oracledb.connect(user=self.username, 
                                         password=self.password, 
                                         dsn=self.dsn)
            print("Connection successful!", self.conn.version)
            return True
        except Exception as e:
            print("Connection failed!")
            return False
        
    def configure_knowledge_base(self):
        if self.conn == None:
            print("Have you called connect()? self.conn is empty.")
            return False

        print("Saving data into DB...")
        for i in range(len(self.docs_dict)):
            print("Saving {}-th data".format(i))
            self.knowledge_base = \
            OracleVS.from_documents(self.docs_dict[i], 
                                    self.upstage_embeddings, 
                                    client=self.conn, 
                                    table_name="text_embedding_{}".format(i), 
                                    distance_strategy=DistanceStrategy.DOT_PRODUCT)
        
    def get_vector_store(self):
        vector_store = OracleVS(client=self.conn, 
                                embedding_function=self.upstage_embeddings, 
                                table_name="text_embeddings2", 
                                distance_strategy=DistanceStrategy.DOT_PRODUCT)
        return vector_store

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

class Tools:
    
    def __init__(self):
        self.tools = [pdf_search, internet_search]
        
    @tool
    def pdf_search(self, query: str)->str:
        """Query for the pdf search, given by the user.
        If the user asks answer for the question, primarily the answer is searched in the pdf.
        """
        return DataLoader.docs_dict

    @tool
    def internet_search(self, query: str)->str:
        """Query for the internet search, in search engine like GOOGLE.
        If the user asks answer for the general question searched for the internet.
        """
        return tavily.search(query=query)
    
    def add_tools(self, llm):
        return llm.bind_tools(self.tools)
    
    def call_tool_func(tool_call):
        tool_name = tool_call["name"].lower()
        if tool_name not in globals():
            print("Tool not found", tool_name)
            return None
        selected_tool = globals()[tool_name]
        
        return selected_tool.invoke(tool_call["args"])

def main():
    data_loader = DataLoader("data/")
    docs_dict = data_loader.load_data()

    oracle_db = OracleDB()
    conn = oracle_db.connect()
    if(oracle_db.configure_knowledge_base()):
        vector_store = oracle_db.get_vector_store()
    retriever = vector_store.as_retriever()

    llm_invoker = LLMInvoker(retriever)
    
    tool = Tools()
    llm_invoker.llm = tool.add_tools(llm_invoker.llm)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = llm_invoker.invokellm(user_input, """Answer the question based only on the following context:
                                                        {context} 
                                                        Question: {question} 
                                                     """)
        if llm_invoker.groundedness_check(response):
            print("Bot:", response)

        else:
            for _ in range(3):
                tool_calls = llm_invoker.llm.invoke(user_input).tool_calls
                if tool_calls:
                    break
            if not tool_calls:
                print("I'm sorry, I don't have an answer for that.")
                break
            
            context = ""
            for tool_call in tool_calls:
                context += str(tool.call_tool_func(tool_call))

            response = llm_invoker.invokellm(user_input, """Answer the question based only on the following context:
                                                        {context} 
                                                        Question: {question} 
                                                     """)
            print("Bot:", response)


    OracleDBIndex.create_index(conn, vector_store)

if __name__ == "__main__":
    main()