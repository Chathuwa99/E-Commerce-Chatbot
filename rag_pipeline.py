import os
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_qa_pairs(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    qa_blocks = [q.strip() for q in raw.split("Q: ") if q.strip()]
    documents = []

    for block in qa_blocks:
        if "A:" not in block:
            continue
        question, answer = block.split("A:", 1)
        question = question.strip()
        answer = answer.strip()
        qa_text = f"Q: {question}\nA: {answer}"
        documents.append(Document(page_content=qa_text))
    return documents

def create_vector_store(documents, persist_directory="db"):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
        print(f"🗑️ Deleted old vector DB folder '{persist_directory}' for fresh start.")

    vectordb = Chroma.from_documents(documents, embedding=embedding_function, persist_directory=persist_directory)
    vectordb.persist()
    print(f"✅ Created new vector DB and saved at '{persist_directory}'.")
    return vectordb

def chatbot_loop(vectordb, threshold=0.7):
    print("\n🗣️ Ask your question (type 'exit' to quit):")
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break
        if not query:
            continue

        # Get docs with similarity scores
        results = vectordb.similarity_search_with_score(query, k=1)
        if not results:
            print("🤖 Sorry, I don't have an answer for that.\n")
            continue

        doc, score = results[0]
        # Uncomment below to debug similarity scores
        # print(f"DEBUG: similarity score = {score}")

        if score < threshold:
            print("🤖 Sorry, I don't have an answer for that.\n")
            continue

        content = doc.page_content
        if "Q:" in content and "A:" in content:
            answer = content.split("A:", 1)[1].strip()
            print(f"🤖 {answer}\n")
        else:
            print("🤖 Sorry, answer not found.\n")

def main():
    data_path = "data/ecommerce_faq.txt"  # Path to your FAQ text file
    persist_dir = "db"

    documents = load_qa_pairs(data_path)
    vectordb = create_vector_store(documents, persist_dir)
    chatbot_loop(vectordb)

if __name__ == "__main__":
    main()
