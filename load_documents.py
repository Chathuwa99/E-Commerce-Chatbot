from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the document with proper encoding
loader = TextLoader("data/ecommerce_faq.txt", encoding="utf-8")
documents = loader.load()

# Split into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# Print sample chunks to verify
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk.page_content)
