import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain 
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
import jieba

base_dir = os.path.dirname(__file__)
# 资料文本目录
resource_knowlege_txt_dir = os.path.join(base_dir, "text")
jieba_dir = os.path.join(base_dir, "jieba")
vector_dir = os.path.join(base_dir, "vectory")

openai_key = "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"
os.environ["OPENAI_API_KEY"] = openai_key
# 分词
def _cut_words():
    # 分词
    for dir in os.scandir(resource_knowlege_txt_dir):
        if dir.is_file():
            with open(os.path.join(resource_knowlege_txt_dir, dir.name), encoding="utf-8") as f:
                data = f.read()

                words_data = " ".join([w for w in list(jieba.cut(data))])
                if not os.path.exists(jieba_dir):
                    os.mkdir(jieba_dir, 0o755)

                words_file = os.path.join(jieba_dir, dir.name)
                with open(words_file, "w") as f:
                    f.write(words_data)

# 初始化知识库
def init_knowlege():
    _cut_words()
    loader = DirectoryLoader(jieba_dir, glob="**/*.txt")
    docs = loader.load()

    text_splitter = TokenTextSplitter()
    doc_texts = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    # 向量化后保存到文件
    vectordb = Chroma.from_documents(doc_texts, embeddings, persist_directory=vector_dir).as_retriever()
    return vectordb

def instance_chain(vectordb):
    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=.8, model_name="gpt-3.5-turbo"), vectordb)
    return chain

if __name__ == "__main__":
    # 初始化知识库
    vectordb = init_knowlege()

    # chain工具
    chain = instance_chain(vectordb)

    result = chain({"question": "人民币超过美元", "chat_history": []})
    print(result)