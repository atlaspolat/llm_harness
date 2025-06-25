from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration
import torch
import re


# Find the think token ID for parsing
think_token_id = 151668  # </think> token for Qwen3

dataset = load_dataset("AtlasPolat/tyt2024", streaming=False)
print("Dataset loaded successfully!")
print(f"Dataset info: {dataset}")

image_dataset = load_dataset("AtlasPolat/tyt2024_images", streaming=False)
print("Image dataset loaded successfully!")


# output an ,mage to a test file
image_sample = image_dataset["train"][0]["image"] # type: ignore
image_sample.save("sample_image.png")



questions = list(dataset["train"])[95:105] # type: ignore

    
# load the model
model_path = "models/Qwen/Qwen3-8B"  # Path to your local model

gpu_id = 0  # Specify the GPU ID you want to use
    
    # Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{gpu_id}"},  # Force all layers to this specific GPU
        trust_remote_code=True
    )

print(f"[GPU {gpu_id}] Model loaded! Starting to process questions from queue...")


# Define the system prompt
turkish_instruction = "Parçaya ve soruya göre hangi seçenek doğrudur? Cevabınız sadece seçeneğin indeksine karşılık gelen tek bir rakam (0, 1, 2, 3 veya 4) olmalıdır."
    
system_prompt_content = """You are an AI assistant. That would help users answer questions based on provided passages and choices. The language of the questions and passages is Turkish. And you should provide your answer as a single digit corresponding to the index of the correct choice.
""" + turkish_instruction + """
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


Or you can also call certain tools after thinking as the output here is the available tools you can use:

Available Tools:

imageqa: This tool is used for answering questions based on images. It requires an image code and a question text.
There is the image code: <IMG-xxx> which is a placeholder for the image you will use in your question. The image codes are provided in the dataset, and you can use them to refer to specific images.



How to use the tools:

ToolCall: {{ "tool": "imageqa",
            "args": {{
                "image_code": "<IMG-002>",
                 "questions": "<question_text>"   }}
}}

"""


# image model loading

image_model_path = "models/Salesforce/blip2-flan-t5-xl"  # Path to your local image model

# load the model to the specified GPU
image_processor = Blip2Processor.from_pretrained(image_model_path, trust_remote_code=True)
image_model = Blip2ForConditionalGeneration.from_pretrained(
        image_model_path,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{1}"},  # Force all layers to this specific GPU
        trust_remote_code=True
    )
print(f"[GPU {1}] Image model loaded! Ready to process image-based questions...")



def imageqa_tool_call(image_code, question):
    """
    Simulates a tool call for image-based question answering.
    
    Args:
        image_code (str): Base64 encoded image data.
        question (str): The question to be answered.
        choices (list): List of answer choices.
    
    Returns:
        str: The index of the correct answer choice.
    """
    
    return f""" There is nothing in the image, so I cannot answer the question based on it."""

   
def main():

    for i, question in enumerate(questions):
        print(f"Processing question {i + 1} of {len(questions)}...")
        
        #construct the prompt
        prompt_parts = []

        question_text = question.get("question", "")
        passage = question.get("passage", "")
        choices = question.get("choices", [])

        # Construct the prompt
        prompt_parts = []

        prompt_parts.append(f"Passage: {passage}")
        prompt_parts.append(f"Question: {question_text}")
        prompt_parts.append("Choices:")

        for idx, choice in enumerate(choices):
            prompt_parts.append(f"{idx}. {choice.strip()}")

        prompt_parts.append("\nBased on the passage and question, which choice is correct? Your answer should be only a single digit: 0, 1, 2, 3, or 4, corresponding to the choice index.")

        full_prompt = "\n".join(prompt_parts)

        while True:
                messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": full_prompt}
        ]
                text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
                model_inputs = tokenizer([text_input], return_tensors="pt").to(f"cuda:{gpu_id}")

                # Generate response
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=16384,  # Reduced for better performance
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
                    print(f"[GPU {gpu_id}] Warning: </think> token not found in output.")
                    content_after_thinking = " -1 [No explicit thinking block found]"
                    thinking_content = tokenizer.decode(output_ids_only, skip_special_tokens=True).strip()

                # check if there is a function call in the output
                if "ToolCall:" in content_after_thinking:
                    # Extract the parts of the function call
                    tool_call_match = re.search(r'ToolCall:\s*{([^}]*)}', content_after_thinking)
                    if tool_call_match:
                        tool_call_content = tool_call_match.group(1)
                        print(f"[GPU {gpu_id}] Tool call detected: {tool_call_content}")
                        # Here you can handle the tool call if needed
                        # For now, we will just print it
                        # turn it into a dict using jason.loads
                        import json
                        tool_call_dict = {}
                        try:
                            tool_call_dict = json.loads(tool_call_content)
                        except json.JSONDecodeError as e:
                            print(f"[GPU {gpu_id}] Error decoding tool call content: {e}")

                        if tool_call_dict.get("tool") == "imageqa":
                            image_code = tool_call_dict.get("args", {}).get("image_code", "")
                            question_text = tool_call_dict.get("args", {}).get("questions", "")
                            
                            # Simulate the imageqa tool call
                            answer = imageqa_tool_call(image_code, question_text)
                            print(f"[GPU {gpu_id}] ImageQA Tool Call Answer: {answer}")

                            # add the content beforethe tool call to the prompt parts
                            prompt_parts.append(thinking_content)

                            # add the image numebr and the question to the prompt parts
                            prompt_parts.append(f"Image Code: {image_code}")
                            prompt_parts.append(f"Question: {question_text}")
                            # add the tool call answer to the prompt parts
                            prompt_parts.append(f"Answer: {answer}")
                        else:
                            print(f"[GPU {gpu_id}] Unknown tool call detected: {tool_call_dict.get('tool')}")
                    else:
                        print(f"[GPU {gpu_id}] No valid tool call found in output.")


                # Extract the answer
                match = re.search(r'\b([0-4])\b', content_after_thinking)
                if not match:
                    match = re.search(r'([0-4])', content_after_thinking)

                if match:
                    parsed_answer = int(match.group(1))
                else:
                    parsed_answer = -1

                print(f"[GPU {gpu_id}] Question {question.get("question_number")} Proccesed- Parsed: {parsed_answer}, Correct: {question.get('answer')}")


            
            


    
   
if __name__ == "__main__":
    main()
