import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Better text splitter
from langchain_community.vectorstores import FAISS  # Use FAISS instead of Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

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

# ==== 4. Set up RAG chain with conversation memory ====
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Limit to top 3 most relevant chunks
)

# Create conversation memory (deprecated but keeping for now)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="result"  # Changed from "answer" to "result"
)

# Create a custom prompt template that includes chat history
prompt_template = """Use the following pieces of context and conversation history to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Chat History: {chat_history}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "chat_history", "question"]
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True  # This helps debug what context is being used
)

# ==== 5. Question-answering loop ====
while True:
    user_input = input("Type 'exit' to quit or ask a question: ")
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break
      # Get chat history for the prompt
    chat_history_str = ""
    chat_history = memory.chat_memory.messages
    for i in range(0, len(chat_history), 2):
        if i + 1 < len(chat_history):
            chat_history_str += f"Human: {chat_history[i].content}\n"
            chat_history_str += f"Assistant: {chat_history[i+1].content}\n"
    
    # Get context from retriever
    docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Use invoke with all required variables
    result = rag_chain.invoke({
        "query": user_input,
        "context": context,
        "chat_history": chat_history_str,
        "question": user_input
    })
    
    print("\n" + "="*50)
    print("QUESTION:", user_input)
    print("="*50)
    print("ANSWER:", result["result"])
    print("="*50)
    
    # Manually add to conversation memory
    memory.save_context({"input": user_input}, {"result": result["result"]})
    
    # Show conversation history
    print("CONVERSATION HISTORY:")
    chat_history = memory.chat_memory.messages
    for i in range(0, len(chat_history), 2):
        if i + 1 < len(chat_history):
            print(f"Q: {chat_history[i].content}")
            print(f"A: {chat_history[i+1].content}")
    print("="*50)
    
    # Optionally show source documents for debugging
    if result.get("source_documents"):
        print("SOURCES USED:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"Source {i+1}: {doc.page_content[:100]}...")
        print("="*50)