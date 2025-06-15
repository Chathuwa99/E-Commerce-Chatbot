# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# import os

# # Path to your Q&A data file
# data_path = "data/ecommerce_faq.txt"

# # Vector DB directory
# persist_directory = "db"

# # Load your Q&A file
# loader = TextLoader(data_path, encoding="utf-8")
# documents = loader.load()

# # Split text into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=50
# )
# chunks = text_splitter.split_documents(documents)

# # Load HuggingFace Embeddings
# embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Check if DB already exists (to append) or needs creation
# if os.path.exists(persist_directory):
#     vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
#     vectordb.add_documents(chunks)
#     print("📌 Appended new data to existing Chroma vector DB.")
# else:
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_function,
#         persist_directory=persist_directory
#     )
#     print("✅ Created new Chroma vector DB with initial data.")

# # Save changes
# vectordb.persist()
# print("💾 Vector DB saved successfully.")

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
import os

# Step 1: Read file and separate each Q&A pair
file_path = "data/ecommerce_faq.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_data = f.read()

# Step 2: Split based on "Q: " — ensures each Q&A is isolated
qa_pairs = [q.strip() for q in raw_data.split("Q: ") if q.strip()]
documents = []

for pair in qa_pairs:
    question, answer = pair.split("A:", 1)
    qa_text = f"Q: {question.strip()}\nA: {answer.strip()}"
    documents.append(Document(page_content=qa_text))

# Step 3: Set up embedding and vector store
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "db"

# Step 4: Append or create
if os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    vectordb.add_documents(documents)
    print("📌 Appended new Q&A entries.")
else:
    vectordb = Chroma.from_documents(documents, embedding=embedding_function, persist_directory=persist_directory)
    print("✅ Created new vector store.")

vectordb.persist()
print("💾 Vector DB updated successfully.")
