import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

loader = PyPDFLoader("moonwalking.pdf")
documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)
# print("Number of chunks:", len(docs))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768
)

# batch_size = 15
# for i in range(0, len(docs), batch_size):
#     batch = docs[i:i + batch_size]
#     if i == 0:
#         vectorstore = FAISS.from_documents(batch, embeddings)
#     else:
#         vectorstore.add_documents(batch)
#     print(f"Uploaded batch {i//batch_size + 1}. Sleeping to save quota...")
#     time.sleep(35)
# vectorstore.save_local("faiss_index")
# print("Vector store created and stored locally")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer the question.\n\n{context}"),
    ("human", "{input}")
])

# Build the chain using LCEL
document_chain = create_stuff_documents_chain(model, prompt)
qa_chain = create_retrieval_chain(vectorstore.as_retriever(), document_chain)

query = ""

response = qa_chain.invoke({"input": query})
print("Answer:", response["answer"])

while True:
    query = input("Ask a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    response = qa_chain.invoke({"input": query})
    print("Answer:", response["answer"])