import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ==== 1. Load your local Hugging Face model with GPU support ====
MODEL_PATH = "models/Qwen/Qwen3-8B"  # folder with the model files

# Find the first available GPU
available_gpus = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_properties(i).total_memory > 0]
if not available_gpus:
    raise RuntimeError("No available GPUs found. Please check your setup.")

print(f"Available GPUs: {available_gpus}")

first_gpu = available_gpus[0]
device_name = f"cuda:{first_gpu}"


print(f"Using GPU {first_gpu}: {torch.cuda.get_device_name(first_gpu)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# OR Option 2: Manual device placement (if you prefer explicit control)
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

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

# ==== 3. Create embeddings and vector store ====
# Assign embedding model to the same GPU
embedding_model = HuggingFaceEmbeddings(
    model_name="models/all-MiniLM-L6-v2",
    model_kwargs={'device': device_name}  # Assign to first available GPU
)

vectorstore = Chroma.from_documents(split_docs, embedding=embedding_model)

# ==== 4. Set up RAG chain ====
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ==== 5. Ask your question ====
question = input("Ask a question: ")
response = rag_chain.run(question)
print("\nAnswer:", response)