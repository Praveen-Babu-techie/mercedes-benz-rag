import os
import json
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from flask import Flask, request, Response
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.llms import Cohere
from langchain_cohere import CohereRerank
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache


application = Flask(__name__)
CORS(application)

db_path = "AMG G-Class Owners Manual_FAISS_DB"

load_dotenv()
api_key =  os.environ["api_key"]
cohere_api_key = os.environ["cohere_api_key"] 

embeddings = OpenAIEmbeddings(api_key=api_key)
local_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
retriever = local_db.as_retriever(search_type="mmr", search_kwargs={"k": 20})
# llm = Cohere(temperature=0,cohere_api_key=cohere_api_key)
compressor = CohereRerank(cohere_api_key=cohere_api_key)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
set_llm_cache(InMemoryCache())
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=api_key)


@application.route("/", methods=["GET"])
@cross_origin()
def hello():
    return "Hello World"


@application.route("/assistant", methods=["POST"])
@cross_origin()
def home():
    try:
        data_string = request.get_data()
        data = json.loads(data_string)
        query = data.get("query")
        compressed_docs = compression_retriever.invoke(query)
        message = ""
        for document in compressed_docs:
            content = document.page_content
            message += content
        system_template = SystemMessagePromptTemplate.from_template(
            "You're an excellent Answer provider for a QnA used as a Car Assistant, you will provide brief summary answers in two sentence only from the content provided to you strictly , You can understand the content even it is unformatted. \nContent\n"
            + message
        )
        user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
        template = ChatPromptTemplate.from_messages([system_template, user_template])
        chain = LLMChain(llm=llm, prompt=template)
        result = chain.invoke({"user_prompt": query})
        return {"response": result["text"]}
    except Exception as e:
        return Response(
            f"bad request! - {e} ",
            400,
        )


if __name__ == "__main__":
    application.run(debug=True)
