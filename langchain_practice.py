import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Better text splitter
from langchain_community.vectorstores import FAISS  # Use FAISS instead of Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ==== 1. Load your local Hugging Face model with GPU support ====
MODEL_PATH = "models/Qwen/Qwen3-8B"

# Find the first available GPU
available_gpus = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_properties(i).total_memory > 0]
if not available_gpus:
    raise RuntimeError("No available GPUs found. Please check your setup.")

print(f"Available GPUs: {available_gpus}")

first_gpu = available_gpus[0]
device_name = f"cuda:{first_gpu}"

print(f"Using GPU {first_gpu}: {torch.cuda.get_device_name(first_gpu)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
     MODEL_PATH,
     torch_dtype=torch.float16,
     low_cpu_mem_usage=True
 ).to(device_name)

pipe = pipeline(
     "text-generation",
     model=model,
     tokenizer=tokenizer,
     max_new_tokens=512,
     device=device_name,
 )

llm = HuggingFacePipeline(pipeline=pipe)

# ==== 2. Load and split your document ====
loader = TextLoader("my_docs.txt")
documents = loader.load()

# Use RecursiveCharacterTextSplitter for better chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
split_docs = text_splitter.split_documents(documents)

print(f"Created {len(split_docs)} chunks")
for i, doc in enumerate(split_docs[:3]):
    print(f"Chunk {i+1} size: {len(doc.page_content)} characters")

# ==== 3. Create embeddings and vector store ====
embedding_model = HuggingFaceEmbeddings(
    model_name="models/all-MiniLM-L6-v2",
    model_kwargs={'device': device_name}
)

# Use FAISS instead of Chroma
vectorstore = FAISS.from_documents(split_docs, embedding=embedding_model)

# ==== 4. Set up RAG chain ====
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)



while True:
    # Check if the user wants to exit
    user_input = input("Type 'exit' to quit or sk a question: ")
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break

    
    response = rag_chain.run(user_input)
    print("\nAnswer:", response)