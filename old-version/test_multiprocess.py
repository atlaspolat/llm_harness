from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def print_gpu_diagnostics():
    """Print GPU diagnostics for the current process"""
    print("="*60)
    print(f"Process ID: {os.getpid()}")
    print("GPU DIAGNOSTICS")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs visible: {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print("="*60)

def load_model_and_infer(args):
    """Worker function that runs in a separate process for each GnPU"""
    device_id, prompts, model_path = args
    
    # Print diagnostics for this process
    print_gpu_diagnostics()
    
    # Set the current device for this process
    torch.cuda.set_device(device_id)
    
    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    
    print(f"[GPU {device_id}] Loading model on process {os.getpid()}...")
    
    # Load the tokenizer and model on the specific GPU
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use float16 for better performance
        device_map={"": f"cuda:{device_id}"},  # Force all layers to this specific GPU
        trust_remote_code=True
    )
    
    print(f"[GPU {device_id}] Model loaded! Processing {len(prompts)} prompts...")
    
    # Process each prompt
    results = []
    for i, prompt in enumerate(prompts):
        print(f"[GPU {device_id}] Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(f"cuda:{device_id}")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,  # Reduced for faster processing
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        # Store result
        result = {
            'prompt': prompt,
            'thinking': thinking_content,
            'content': content,
            'gpu_id': device_id
        }
        results.append(result)
        
        # Save individual result to file
        with open(f"output_gpu_{device_id}.txt", "a") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Thinking Content: {thinking_content}\n")
            f.write(f"Content: {content}\n")
            f.write("\n" + "="*50 + "\n\n")
        
        print(f"[GPU {device_id}] Completed prompt {i+1}: {content[:100]}...")
    
    print(f"[GPU {device_id}] Finished all {len(prompts)} prompts on process {os.getpid()}!")
    return results

def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Print initial diagnostics
    print_gpu_diagnostics()
    
    model_name = "Qwen/Qwen3-8B"
    model_path = f'/kuacc/users/apolat21/lm_models/{model_name}'
    
    # Test prompts
    prompts = [
        "What is the smallest prime number?",
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
        "What is the future of artificial intelligence?",
        "Why are we drowning in plastic?"
    ]
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        return
    
    # Distribute prompts among GPUs
    prompts_per_gpu = [[] for _ in range(num_gpus)]
    for i, prompt in enumerate(prompts):
        prompts_per_gpu[i % num_gpus].append(prompt)
    
    # Print distribution
    for i in range(num_gpus):
        print(f"GPU {i} will process {len(prompts_per_gpu[i])} prompts")
    
    # Clear existing output files
    for i in range(num_gpus):
        if os.path.exists(f"output_gpu_{i}.txt"):
            os.remove(f"output_gpu_{i}.txt")
    
    # Prepare arguments for each process
    args_list = []
    for i in range(num_gpus):
        args_list.append((i, prompts_per_gpu[i], model_path))
    
    print("Starting parallel inference using ProcessPoolExecutor...")
    start_time = time.time()
    
    # Use ProcessPoolExecutor for true parallel execution
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all GPU tasks
        futures = []
        for i, args in enumerate(args_list):
            print(f"Submitting GPU {i} task with {len(args[1])} prompts to separate process")
            future = executor.submit(load_model_and_infer, args)
            futures.append(future)
        
        # Wait for all tasks to complete and collect results
        all_results = []
        for i, future in enumerate(futures):
            try:
                results = future.result()  # This will block until the task completes
                all_results.extend(results)
                print(f"GPU {i} completed its tasks! Processed {len(results)} prompts")
            except Exception as e:
                print(f"GPU {i} encountered an error: {e}")
                import traceback
                traceback.print_exc()
    
    end_time = time.time()
    print(f"All GPUs completed! Total time: {end_time - start_time:.2f} seconds")
    print(f"Total prompts processed: {len(all_results)}")
    
    # Save combined results
    with open("combined_results.txt", "w") as f:
        for result in all_results:
            f.write(f"GPU {result['gpu_id']} - Prompt: {result['prompt']}\n")
            f.write(f"Thinking: {result['thinking']}\n")
            f.write(f"Content: {result['content']}\n")
            f.write("\n" + "="*80 + "\n\n")

if __name__ == "__main__":
    main()
