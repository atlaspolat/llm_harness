from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re # For parsing the final answer
import os

dataset = load_dataset("AtlasPolat/yks2024", streaming=False)

df = pd.DataFrame(dataset)


print(df.size)
print(df.head())


model_name = "Qwen/Qwen3-8B" # Path to the model, can be a local path or a Hugging Face model hub path


# Check if the model is already downloaded 
# If not, it will download it from the Hugging Face model hub


model_path = f'/kuacc/users/apolat21/lm_models/{model_name}'  # Adjust this path if needed

folder_path = f'/kuacc/users/apolat21/lm_harness_results/{model_name}'



# check if the model already exists in the model_path
if os.path.exists(model_path):
        
        # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
    )
    
else:
                # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    )
        print(f"Model does not exist at {model_path}. Saving model and tokenizer...")
        os.makedirs(model_path, exist_ok=True)
        print(f"Saving model to {model_path}...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved successfully!")



results_list = []
think_token_id = 151668 # </think> token ID for Qwen/Qwen3-8B

# Assuming your questions are in df['train']
# Adjust this if your data structure is different
if 'train' in df:
    question_data = df['train']
    print(f"Processing {len(question_data)} questions from df['train']...")
elif 'train' in dataset: # Fallback if df['train'] was not created as expected
    try:
        question_data = [item for item in dataset['train']] # Ensure it's a list
        print(f"Processing {len(question_data)} questions from dataset['train']...")
    except TypeError:
        print("dataset['train'] is not iterable. Please ensure it's loaded correctly.")
        question_data = []
else:
    print("Could not find 'train' data in `df` or `dataset`. Please check your data loading steps.")
    question_data = []

# Define the new system prompt with the Turkish instruction and few-shot example
turkish_instruction = "Parçaya ve soruya göre hangi seçenek doğrudur? Cevabınız sadece seçeneğin indeksine karşılık gelen tek bir rakam (0, 1, 2, 3 veya 4) olmalıdır."

system_prompt_content = f"""You are an AI assistant. That would help users answer questions based on provided passages and choices. The language of the questions and passages is Turkish. And you should provide your answer as a single digit corresponding to the index of the correct choice.
{turkish_instruction}
Analyze the question and provide your thinking process before giving the final answer as a single digit.

Here is an example of the input format you will receive and the thinking process/output format you should follow:

Input Example (User will provide this structure):
Passage: Kitap Sanat, edebiyat ve eleştirinin kökeninin yakın dönemlere dayandığını iddia ederken resim, şiir ve müzik türlerinin izlerine daha önceki çağlarda rastlanmadığını değil; bu türlere bakışımızın önceki dönemlerden farklı olduğunu ileri sürüyorum. Söz gelimi İlyada destanı 2000’li yıllarda Arkaik Çağ’dakinden çok farklı işlevler görür. Modern okurlar için bir başyapıt olarak Batı edebiyatında önemli bir yeri vardır. Ama Antik Yunan’da yaşayanlar, onun edebiyat olduğunu düşünemezlerdi çünkü kavramsal olarak henüz böyle bir sınıflama yoktu. Bu epik şiir, kurmaca olması yönüyle benzersiz bir yazın tarzı payesi almak şöyle dursun, toplumsal hayatla sıkı sıkıya bütünleşmişti; törenlerde okunuyor, öğretiliyor ve sık sık hukuksal anlaşmazlıkların karara bağlanmasında kullanılıyordu.
Question: Bu parçaya göre Antik Yunan’da İlyada destanının günümüzdeki gibi bir edebî yapıt olarak görülmemesinin nedeni aşağıdakilerden hangisidir?
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


for i, item in enumerate(question_data):
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
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Crucial for Qwen3-8B thinking mode
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        # Conduct text completion
        # max_new_tokens might need to be generous to allow for thinking.
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=16384, # Increased to allow for thinking
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

        output_ids_only = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        thinking_content = ""
        content_after_thinking = ""
        parsed_answer = -1

        try:
            # Find the last occurrence of the </think> token
            # rindex finds the first occurrence from the end
            index_of_think_end = len(output_ids_only) - output_ids_only[::-1].index(think_token_id)

            thinking_output_ids = output_ids_only[:index_of_think_end]



            thinking_content = tokenizer.decode(thinking_output_ids, skip_special_tokens=False).strip()
            # And the content part (after </think>)
            content_after_thinking_ids = output_ids_only[index_of_think_end:]
            content_after_thinking = tokenizer.decode(content_after_thinking_ids, skip_special_tokens=True).strip()

        except ValueError:
            # </think> token not found, assume the whole output is the content
            print(f"Warning: </think> token (ID {think_token_id}) not found in output for question {i+1}.")
            content_after_thinking = " -1 [No explicit thinking block found]"
            thinking_content = tokenizer.decode(output_ids_only, skip_special_tokens=True).strip()

        # Attempt to extract the first digit from the content_after_thinking
        # Using regex to find the first single digit
        match = re.search(r'\b([0-4])\b', content_after_thinking) # Look for a single digit 0-4 as a whole word
        if not match: # If not found as a whole word, try any digit
            match = re.search(r'([0-4])', content_after_thinking)

        if match:
            parsed_answer = int(match.group(1))
        else:
            parsed_answer = -1 # Default if no number 0-4 is found

        print(f"\n--- Question {i+1} ---")
        # print(f"Prompt:\n{full_prompt}") # Can be verbose
        print(f"Thinking Content:\n{thinking_content}")
        print(f"Content After Thinking (raw answer part):\n{content_after_thinking}")
        print(f"Parsed Answer (index): {parsed_answer}")
        print(f"Actual Answer (index): {item.get('answer')}")

        results_list.append({
            "question_number_in_dataset": item.get("question_number"),
            "section": item.get("section"),
            "question": question_text,
            "thinking_content": thinking_content,
            "raw_content_after_thinking": content_after_thinking,
            "parsed_answer_index": parsed_answer,
            "correct_answer_index": item.get("answer"),
            "correct": "true" if parsed_answer == item.get("answer") else "false"
        })

    except Exception as e:
        print(f"Error processing question {i+1}: {e}")
        results_list.append({
            "question_number_in_dataset": item.get("question_number"),
            "section": item.get("section"),
            "question": question_text,
            "error": str(e)
        })

    # Optional: break after a few questions for testing
    if i >= 5: # Process only the first question for quick test
         print("Stopping early for testing.")
         break

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results_list)
print("\n\n--- Batch Processing Complete ---")
if not results_df.empty:
    print(results_df.head())
    # Display relevant columns if the DataFrame is large
    if len(results_df.columns) > 5:
        print("\nKey columns from results:")
        print(results_df[['question_number_in_dataset', 'parsed_answer_index', 'correct_answer_index', 'thinking_content']].head())
        #save the results to a CSV file

else:
    print("No results were processed.")

# You can now save or further analyze results_df
# For example: results_df.to_csv("model_answers_qwen3_8b_thinking.csv", index=False)

results_df.to_csv(f"{folder_path}/model_answers_qwen3_8b_thinking.csv", index=False)


# lets print the results_df by comparing the parsed_answer_index and correct_answer_index


dict_results = {}

for index, row in results_df.iterrows():
    # If the question is correctly answere
    # if correcct add to the dict_results the section make it one if the section is not in the dict_results if it is in the dict_results add one to the section
    if row['parsed_answer_index'] == row['correct_answer_index']:
        if row['section'] not in dict_results:
            dict_results[row['section']] = 1
        else:
            dict_results[row['section']] += 1