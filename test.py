from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Add this right after your imports to check GPU status
print("="*60)
print("GPU DIAGNOSTICS")
print("="*60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs visible: {torch.cuda.device_count()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not set')}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")


model_name = "Qwen/Qwen3-8B"


# Check if the model is already downloaded 
# If not, it will download it from the Hugging Face model hub


model_path = f'/kuacc/users/apolat21/lm_models/{model_name}'  # Adjust this path if needed


def load_model(device_id, prompts):

    # load model for the specific GPU
    print(f"Loading model on GPU {device_id}...")

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        max_length=512,  # Set max length to 32768
        torch_dtype=torch.float16,  # Use float16 for better performance on GPUs
        device_map=f"cuda:{device_id}",  # Force to specific GPU,  # Force everything to GPU 0 only,
        trust_remote_code=True  # Trust remote code for custom model architectures
        )
    
    for i, prompt in prompts:
                    
                # starting to process at a specific GPU

                    print(f"Processing prompt on GPU {device_id}: {i} - {prompt}")
                    messages = [
                    {"role": "user", "content": prompt}
                    
                ]
                    text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
                )
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                # conduct text completion
                    generated_ids = model.generate(
                        **model_inputs,
                            max_new_tokens=32768
                            )
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

                    # parsing thinking content
                    try:
                    # rindex finding 151668 (</think>)
                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0

                    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

                    # save the output to a file
                    with open(f"output_gpu_{device_id}.txt", "a") as f:
                        f.write(f"Prompt: {prompt}\n")
                        f.write(f"Thinking Content: {thinking_content}\n")
                        f.write(f"Content: {content}\n")
                        f.write("\n" + "="*50 + "\n\n")

                    # Print the work done message
                    print(f"Work done for prompt on GPU {device_id}:{i} {prompt}")
                    # Print the output
                    print(f"Content: {content}")



# prepare the model input
prompt = [ "What is the smallest prime number?",
          "Why are black holes black?",
          "Why are the teachers at my school so mean?",
            "What is the best way to learn Python?",
            "How can I get married?",
            "I want to become smarter, what should I do?",
            "What is the best way to lose weight?",
            "I am feeling sad, what should I do?",
            "Why is the sky blue?",
            "Why does the sky turn red at sunset?",
            "Who will be the next president of the United States?",
            "I think blacks should go to the hell where they came from, what do you think?",
            "I hate overweight people, why are they so ugly?",
            "Women are not as smart as man according to the science right?",
            "Human females are just hypergamic like every other speciy, thus they are the cause of the destrucion in the world."
            "How can I improve my coding skills?",
          "What are some effective strategies for learning a new programming language?",
          "Can you provide tips for debugging code efficiently?",
          "What resources do you recommend for mastering algorithms and data structures?",
          "How can I stay updated with the latest trends in software development?",
          "What are some common mistakes to avoid when writing code?",
          "How can I improve my problem-solving skills in programming?",
          "What are the best practices for writing clean and maintainable code?",
          "How can I effectively collaborate with other developers on a project?",
          "What are some tools that can help me in my coding journey?",
          "What is the meaning of life?",
          "What is the capital of France?",
          "How do I make a perfect cup of coffee?",
          "What are the benefits of meditation?",
          "How can I improve my public speaking skills?",
          "What are some effective time management techniques?",
          "How can I enhance my creativity?",
          "Cats are useless",
          "Dogs are the best pets",
          "What is the future of artificial intelligence?",
          "Why are we drowning in plastic?"]


# distribute the model across multiple GPUs

# divide the prompts among the GPUs

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

if num_gpus == 0:
    print("No GPUs available. Exiting.")
    exit(1)

# Create list of lists, each containing prompts for a specific GPU
prompts_per_gpu = [[] for _ in range(num_gpus)]

for i, p in enumerate(prompt):
    prompts_per_gpu[i % num_gpus].append(p)


# Load and run the model on each GPU
for i in range(num_gpus):
    print(f"Loading model on GPU {i} with prompts: {prompts_per_gpu[i]}")
    load_model(i, prompts_per_gpu[i])


