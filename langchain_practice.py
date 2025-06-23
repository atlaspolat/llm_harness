from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ==== 1. Load your local Hugging Face model ====
MODEL_PATH = "models/Qwen/Qwen3-8B"  # folder with the model files
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# ==== 2. Load and split your document ====
loader = TextLoader("my_docs.txt")  # your file with knowledge
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

# ==== 3. Create embeddings and vector store ====
embedding_model = HuggingFaceEmbeddings(model_name="models/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(split_docs, embedding=embedding_model)

# ==== 4. Set up RAG chain ====
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ==== 5. Ask your question ====
question = input("Ask a question: ")
response = rag_chain.run(question)
print("\nAnswer:", response)