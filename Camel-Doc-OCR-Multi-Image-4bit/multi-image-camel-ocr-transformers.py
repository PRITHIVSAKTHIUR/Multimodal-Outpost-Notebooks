# Install necessary libraries
# !pip install git+https://github.com/huggingface/transformers.git \
#              git+https://github.com/huggingface/accelerate.git \
#              git+https://github.com/huggingface/peft.git \
#              transformers-stream-generator huggingface_hub albumentations \
#              pyvips-binary qwen-vl-utils sentencepiece opencv-python docling-core \
#              python-docx torchvision safetensors matplotlib num2words \

# !pip install xformers requests pymupdf hf_xet spaces pyvips pillow \
#              einops torch fpdf timm av decord bitsandbytes
# #Hold tight, this will take around 1-2 minutes.

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import uuid
import io
from threading import Thread
import requests

# Define model options
MODEL_OPTIONS = {
    "Camel-Doc-OCR-080125": "prithivMLmods/Camel-Doc-OCR-080125",
}

# Define 4-bit quantization configuration
# This config will load the model in 4-bit to save VRAM.
# You can customize these settings as needed.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Preload models and processors into CUDA
models = {}
processors = {}
for name, model_id in MODEL_OPTIONS.items():
    print(f"Loading {name}🤗. This will use 4-bit quantization to save VRAM.")
    models[name] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto"
    )
    processors[name] = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

image_extensions = Image.registered_extensions()

def identify_and_save_blob(blob_path):
    """Identifies if the blob is an image and saves it."""
    try:
        with open(blob_path, 'rb') as file:
            blob_content = file.read()
            try:
                Image.open(io.BytesIO(blob_content)).verify()  # Check if it's a valid image
                extension = ".png"  # Default to PNG for saving
                media_type = "image"
            except (IOError, SyntaxError):
                raise ValueError("Unsupported media type. Please upload a valid image.")

            filename = f"temp_{uuid.uuid4()}_media{extension}"
            with open(filename, "wb") as f:
                f.write(blob_content)

            return filename, media_type

    except FileNotFoundError:
        raise ValueError(f"The file {blob_path} was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the file: {e}")

def qwen_inference(model_name, media_input, text_input=None):
    """Handles inference for the selected model."""
    model = models[model_name]
    processor = processors[model_name]

    if isinstance(media_input, str):
        media_path = media_input
        if media_path.endswith(tuple([i for i in image_extensions.keys()])):
            media_type = "image"
        else:
            try:
                media_path, media_type = identify_and_save_blob(media_input)
            except Exception as e:
                raise ValueError("Unsupported media type. Please upload a valid image.")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": media_type,
                    media_type: media_path
                },
                {"type": "text", "text": text_input},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        # Remove <|im_end|> or similar tokens from the output
        buffer = buffer.replace("<|im_end|>", "")

    return buffer

def download_image(url):
    """Downloads an image from a URL and saves it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get the file extension from the URL
        file_extension = os.path.splitext(url)[1]
        if not file_extension:
            # If no extension, try to guess from content type
            content_type = response.headers.get('content-type')
            if content_type:
                import mimetypes
                file_extension = mimetypes.guess_extension(content_type) or '.jpg'
            else:
                file_extension = '.jpg'

        filename = f"temp_{uuid.uuid4()}{file_extension}"
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def main():
    """
    Main function to run multi-page OCR inference on a list of example images.
    """
    example_images = [
        "https://huggingface.co/spaces/prithivMLmods/Multimodal-VLM-v1.0/resolve/main/images/1.png",
        "https://huggingface.co/spaces/prithivMLmods/POINTS-Reader-OCR/resolve/main/examples/2.jpeg"
    ]

    default_prompt = "Perform OCR on the image precisely."
    model_name = "Camel-Doc-OCR-080125" # Or choose from MODEL_OPTIONS

    for i, image_url in enumerate(example_images):
        print(f"--- Processing Image {i+1} from: {image_url} ---")

        # Download the image
        local_image_path = download_image(image_url)

        if local_image_path:
            # Perform OCR
            print("Performing OCR...")
            try:
                ocr_result = qwen_inference(model_name, local_image_path, default_prompt)
                print("\n--- OCR Result ---")
                print(ocr_result)
                print("--------------------\n")
            except Exception as e:
                print(f"An error occurred during inference: {e}")
            finally:
                # Clean up the downloaded image file
                os.remove(local_image_path)

if __name__ == "__main__":
    main()