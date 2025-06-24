from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch




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



def imageqa_tool_call(image_code):

   
def main():

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


Or you can also call certain tools here is the available tools you can use:

Available Tools:



How to use the tools:

ToolCall: { "tool": "imageqa",
           }

"""
   
if __name__ == "__main__":
    main()
