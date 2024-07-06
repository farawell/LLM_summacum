from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
import os
import oracledb
import dotenv

%load_ext dotenv
%dotenv

username=os.environ["DB_USER"]
password=os.environ["DB_PASSWORD"]
dsn=os.environ["DSN"]

con = oracledb.connect(user=username, password=password, dsn=dsn)

try: 
    conn23c = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!", conn23c.version)
except Exception as e:
    print("Connection failed!")


pdf = ""
layzer = UpstageLayoutAnalysisLoader(pdf, output_type="html")

docs = layzer.load()

text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=1000, chunk_overlap=100, language=Language.HTML
)

splits = text_splitter.split_documents(docs)

