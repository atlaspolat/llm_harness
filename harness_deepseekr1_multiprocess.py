from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re # For parsing the final answer
import os
import torch
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
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not set')}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print("="*60)

def process_questions_on_gpu(args):
    """Worker function that processes a batch of questions on a specific GPU"""
    device_id, question_batch, model_path, folder_path = args
    
    # Print diagnostics for this process
    print_gpu_diagnostics()
    
    # Set the current device for this process
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    
    print(f"[GPU {device_id}] Loading model on process {os.getpid()}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{device_id}"},  # Force all layers to this specific GPU
        trust_remote_code=True
    )
    
    # Find the think token ID for parsing
    think_token_id = 151668  # </think> token for Qwen3
    
    print(f"[GPU {device_id}] Model loaded! Processing {len(question_batch)} questions...")
    
    # Define the system prompt
    turkish_instruction = "Parçaya ve soruya göre hangi seçenek doğrudur? Cevabınız sadece seçeneğin indeksine karşılık gelen tek bir rakam (0, 1, 2, 3 veya 4) olmalıdır."
    
    system_prompt_content = f"""You are an AI assistant. That would help users answer questions based on provided passages and choices. The language of the questions and passages is Turkish. And you should provide your answer as a single digit corresponding to the index of the correct choice.
{turkish_instruction}
Analyze the question and provide your thinking process before giving the final answer as a single digit.

Here is an example of the input format you will receive and the thinking process/output format you should follow:

Input Example (User will provide this structure):
Passage: Kitap Sanat, edebiyat ve eleştirinin kökeninin yakın dönemlere dayandığını iddia ederken resim, şiir ve müzik türlerinin izlerine daha önceki çağlarda rastlanmadığını değil; bu türlere bakışımızın önceki dönemlerden farklı olduğunu ileri sürüyorum. Söz gelimi İlyada destanı 2000'li yıllarda Arkaik Çağ'dakinden çok farklı işlevler görür. Modern okurlar için bir başyapıt olarak Batı edebiyatında önemli bir yeri vardır. Ama Antik Yunan'da yaşayanlar, onun edebiyat olduğunu düşünemezlerdi çünkü kavramsal olarak henüz böyle bir sınıflama yoktu. Bu epik şiir, kurmaca olması yönüyle benzersiz bir yazın tarzı payesi almak şöyle dursun, toplumsal hayatla sıkı sıkıya bütünleşmişti; törenlerde okunuyor, öğretiliyor ve sık sık hukuksal anlaşmazlıkların karara bağlanmasında kullanılıyordu.
Question: Bu parçaya göre Antik Yunan'da İlyada destanının günümüzdeki gibi bir edebî yapıt olarak görülmemesinin nedeni aşağıdakilerden hangisidir?
Choices:
0. Sözlü kültür geleneğinden beslendiği için toplumun ürünü olarak algılanması
1. Sosyal yaşamda bir amaca hizmet ettiği için faydacı yaklaşımla incelenmesi
2. Hukuki metinlerin boşluğunu doldurduğu için estetikyönüne odaklanılmaması
3. Yazınsal metin kategorisi oluşmadığından bu yönünün değerlendirilememesi
4. İçeriğinin, yazıldığı dönemin sosyal ve siyasal işleyişine yönelik tasarlanması

Your Output Example (after your thinking process, which should be enclosed in <think>...</think> tags if the model supports it, or just precede the answer):
<think> The passage states that in Ancient Greece, the concept of 'literature' as a classification did not exist. Therefore, the Iliad could not be seen as a literary work in the way modern readers see it. This directly corresponds to the idea that a 'literary text category' had not yet been formed. Choice 3 reflects this. </think>
3
"""
    
    results_list = []
    
    for i, item in enumerate(question_batch):
        question_text = item.get("question", "")
        passage = item.get("passage", "")
        choices = item.get("choices", [])

        # Construct the prompt
        prompt_parts = []
        if passage:
            prompt_parts.append(f"Passage: {passage}")
        prompt_parts.append(f"Question: {question_text}")
        prompt_parts.append("Choices:")
        for idx, choice in enumerate(choices):
            prompt_parts.append(f"{idx}. {choice.strip()}")

        prompt_parts.append("\nBased on the passage and question, which choice is correct? Your answer should be only a single digit: 0, 1, 2, 3, or 4, corresponding to the choice index.")

        full_prompt = "\n".join(prompt_parts)

        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": full_prompt}
        ]

        try:
            print(f"[GPU {device_id}] Processing question {i+1}/{len(question_batch)}")
            
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            model_inputs = tokenizer([text_input], return_tensors="pt").to(f"cuda:{device_id}")

            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=4096,  # Reduced for better performance
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                )

            output_ids_only = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            thinking_content = ""
            content_after_thinking = ""
            parsed_answer = -1

            try:
                # Find the last occurrence of the </think> token
                index_of_think_end = len(output_ids_only) - output_ids_only[::-1].index(think_token_id)
                thinking_output_ids = output_ids_only[:index_of_think_end]
                thinking_content = tokenizer.decode(thinking_output_ids, skip_special_tokens=False).strip()
                
                # Content after thinking
                content_after_thinking_ids = output_ids_only[index_of_think_end:]
                content_after_thinking = tokenizer.decode(content_after_thinking_ids, skip_special_tokens=True).strip()

            except ValueError:
                print(f"[GPU {device_id}] Warning: </think> token not found in output for question {i+1}.")
                content_after_thinking = " -1 [No explicit thinking block found]"
                thinking_content = tokenizer.decode(output_ids_only, skip_special_tokens=True).strip()

            # Extract the answer
            match = re.search(r'\b([0-4])\b', content_after_thinking)
            if not match:
                match = re.search(r'([0-4])', content_after_thinking)

            if match:
                parsed_answer = int(match.group(1))
            else:
                parsed_answer = -1

            print(f"[GPU {device_id}] Question {i+1} - Parsed: {parsed_answer}, Correct: {item.get('answer')}")

            results_list.append({
                "question_number_in_dataset": item.get("question_number"),
                "section": item.get("section"),
                "question": question_text,
                "thinking_content": thinking_content,
                "raw_content_after_thinking": content_after_thinking,
                "parsed_answer_index": parsed_answer,
                "correct_answer_index": item.get("answer"),
                "correct": "true" if parsed_answer == item.get("answer") else "false",
                "gpu_id": device_id
            })

        except Exception as e:
            print(f"[GPU {device_id}] Error processing question {i+1}: {e}")
            results_list.append({
                "question_number_in_dataset": item.get("question_number"),
                "section": item.get("section"),
                "question": question_text,
                "error": str(e),
                "gpu_id": device_id
            })

    print(f"[GPU {device_id}] Finished processing {len(question_batch)} questions!")
    return results_list

def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Print initial diagnostics
    print_gpu_diagnostics()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("AtlasPolat/yks2024", streaming=False)
    
    # Access the train split directly
    if 'train' in dataset:
        question_data = list(dataset['train'])  # Convert to list for easier manipulation
        print(f"Loaded {len(question_data)} questions from dataset")
    else:
        print("Could not find 'train' data in dataset. Exiting.")
        return
    
    model_name = "Qwen/Qwen3-8B"
    model_path = f'/kuacc/users/apolat21/lm_models/{model_name}'
    folder_path = f'/kuacc/users/apolat21/lm_harness_results/{model_name}'
    
    # Create output directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        return
    
    # For testing, limit to first 50 questions (remove this line for full processing)
    question_data = question_data[:50]  # Remove this line to process all questions
    
    # Distribute questions among GPUs
    questions_per_gpu = [[] for _ in range(num_gpus)]
    for i, question in enumerate(question_data):
        questions_per_gpu[i % num_gpus].append(question)
    
    # Print distribution
    for i in range(num_gpus):
        print(f"GPU {i} will process {len(questions_per_gpu[i])} questions")
    
    # Prepare arguments for each process
    args_list = []
    for i in range(num_gpus):
        args_list.append((i, questions_per_gpu[i], model_path, folder_path))
    
    print("Starting parallel inference using ProcessPoolExecutor...")
    start_time = time.time()
    
    # Use ProcessPoolExecutor for true parallel execution
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all GPU tasks
        futures = []
        for i, args in enumerate(args_list):
            if len(args[1]) > 0:  # Only submit if there are questions to process
                print(f"Submitting GPU {i} task with {len(args[1])} questions to separate process")
                future = executor.submit(process_questions_on_gpu, args)
                futures.append((i, future))
        
        # Wait for all tasks to complete and collect results
        all_results = []
        for gpu_id, future in futures:
            try:
                results = future.result()  # This will block until the task completes
                all_results.extend(results)
                print(f"GPU {gpu_id} completed its tasks! Processed {len(results)} questions")
            except Exception as e:
                print(f"GPU {gpu_id} encountered an error: {e}")
                import traceback
                traceback.print_exc()
    
    end_time = time.time()
    print(f"All GPUs completed! Total time: {end_time - start_time:.2f} seconds")
    print(f"Total questions processed: {len(all_results)}")
    
    # Convert results to DataFrame and save
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_file = f"{folder_path}/model_answers_qwen3_8b_thinking_multiprocess.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        # Calculate accuracy by section
        dict_results = {}
        correct_count = 0
        total_count = len(results_df)
        
        for index, row in results_df.iterrows():
            if row['parsed_answer_index'] == row['correct_answer_index']:
                correct_count += 1
                section = row['section']
                if section not in dict_results:
                    dict_results[section] = 1
                else:
                    dict_results[section] += 1
        
        print(f"\nOverall Accuracy: {correct_count}/{total_count} = {correct_count/total_count*100:.2f}%")
        print(f"Correct answers by section: {dict_results}")
        
        # Show sample results
        print("\nSample results:")
        print(results_df[['question_number_in_dataset', 'parsed_answer_index', 'correct_answer_index', 'correct', 'gpu_id']].head(10))
    else:
        print("No results were processed.")

if __name__ == "__main__":
    main()
