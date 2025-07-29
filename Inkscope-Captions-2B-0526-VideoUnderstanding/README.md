![9.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/-1klnWCRiSPeNT25L4fCI.png)

# **Inkscope-Captions-2B-0526**

> The **Inkscope-Captions-2B-0526** model is a fine-tuned version of *Qwen2-VL-2B-Instruct*, optimized for **image captioning**, **vision-language understanding**, and **English-language caption generation**. This model was fine-tuned on the `conceptual-captions-cc12m-llavanext` dataset (first 30k entries) to generate **detailed, high-quality captions** for images, including complex or abstract scenes.

> [!note]
Colab Demo : https://huggingface.co/prithivMLmods/Inkscope-Captions-2B-0526/blob/main/Inkscope%20Captions%202B%200526%20Demo/Inkscope-Captions-2B-0526.ipynb

> [!note]
Video Understanding Demo : https://huggingface.co/prithivMLmods/Inkscope-Captions-2B-0526/blob/main/Inkscope-Captions-2B-0526-Video-Understanding/Inkscope-Captions-2B-0526-Video-Understanding.ipynb
---

#### Key Enhancements:

* **High-Quality Visual Captioning**: Generates **rich and descriptive captions** from diverse visual inputs, including abstract, real-world, and complex images.

* **Fine-Tuned on CC12M Subset**: Trained using the **first 30k entries** of the *Conceptual Captions 12M (CC12M)* dataset with the **LLaVA-Next formatting**, ensuring alignment with instruction-tuned captioning.

* **Multimodal Understanding**: Supports detailed understanding of **text+image combinations**, ideal for **caption generation**, **scene understanding**, and **instruction-based vision-language tasks**.

* **Multilingual Recognition**: While focused on English captioning, the model can recognize text in various languages present in the image.

* **Strong Foundation Model**: Built on *Qwen2-VL-2B-Instruct*, offering powerful visual-linguistic reasoning, OCR capability, and flexible prompt handling.

---

### How to Use

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load the fine-tuned model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Inkscope-Captions-2B-0526", torch_dtype="auto", device_map="auto"
)

# Load processor
processor = AutoProcessor.from_pretrained("prithivMLmods/Inkscope-Captions-2B-0526")

# Sample input message with an image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Generate a detailed caption for this image."},
        ],
    }
]

# Preprocess input
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# Generate output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

---

### Buffering Output (Optional for streaming inference)

```python
buffer = ""
for new_text in streamer:
    buffer += new_text
    buffer = buffer.replace("<|im_end|>", "")
    yield buffer
```

---

### **Demo Inference**

![Screenshot 2025-05-27 at 03-59-36 Gradio.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ykPB8Yxk0Z_1WDmSoCjKD.png)
![Screenshot 2025-05-27 at 03-59-53 (anonymous) - output_8dc4ad31-403a-4f59-a483-be2aec11b756.pdf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/tBPdM1iyRf8Fi12urNUbt.png)
  
---

### **Video Inference**

![Screenshot 2025-05-27 at 20-35-30 Video Understanding with Inkscope-Captions-2B-0526.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/LrHNNYV1elysHjAmzOXw3.png)

---

### **Key Features**

1. **Caption Generation from Images:**

   * Transforms visual scenes into **detailed, human-like descriptions**.

2. **Conceptual Reasoning:**

   * Captures abstract or high-level elements from images, including **emotion, action, or scene context**.

3. **Multi-modal Prompting:**

   * Accepts both **image and text** input for **instruction-tuned** caption generation.

4. **Flexible Output Format:**

   * Generates output in **natural language**, ideal for storytelling, accessibility tools, and educational applications.

5. **Instruction-Tuned**:

   * Fine-tuned with **LLaVA-Next style prompts**, making it suitable for interactive use and vision-language agents.

---

## **Intended Use**

**Inkscope-Captions-2B-0526** is designed for the following applications:

* **Image Captioning** for web-scale datasets, social media analysis, and generative applications.
* **Accessibility Tools**: Helping visually impaired users understand image content through text.
* **Content Tagging and Metadata Generation** for media, digital assets, and educational material.
* **AI Companions and Tutors** that need to explain or describe visuals in a conversational setting.
* **Instruction-following Vision-Language Tasks**, such as zero-shot VQA, scene description, and multimodal storytelling.
