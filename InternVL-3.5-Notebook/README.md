# **InternVL-3.5-Notebook**

## Overview

This notebook implements a Gradio-based web interface for the InternVL3.5-2B-Pretrained model, a next-generation open-source multimodal large language model from OpenGVLab. The app allows users to upload images, provide prompts, and generate responses using the model's advanced vision-language capabilities. It supports customizable generation parameters, markdown previews, and PDF export features for the output.

The model integrates Multimodal Continual Pre-Training (CPT), Supervised Fine-Tuning (SFT), and a Cascade Reinforcement Learning (CascadeRL) framework, offering improved reasoning, versatility, and efficiency. Key innovations include the Visual Resolution Router (ViR) for dynamic token scaling and Decoupled Vision-Language Deployment (DvD) for optimized computation.

## Features

- Image upload and prompt input for multimodal queries.
- Real-time processing with the InternVL3.5 model to generate text responses.
- Advanced generation settings: max new tokens, temperature, top-p, top-k, and repetition penalty.
- Markdown preview of the generated output.
- PDF generation with customizable options: font size, line spacing, text alignment, and image size.
- PDF preview gallery and downloadable PDF file.
- Clear all functionality to reset inputs and outputs.

## Requirements

- Python 3.12 or compatible.
- Installed libraries (via pip):
  - gradio
  - transformers
  - torch
  - torchvision
  - pillow
  - fitz (PyMuPDF)
  - reportlab
  - numpy
  - requests
  - spaces (for Hugging Face Spaces integration)

The app uses GPU acceleration if available (CUDA recommended for performance).

## Setup

1. Clone or download the notebook/code.
2. Install dependencies:
   ```
   pip install gradio transformers torch torchvision pillow pymupdf reportlab numpy requests spaces
   ```
3. Ensure the model is accessible from Hugging Face: `OpenGVLab/InternVL3_5-2B-Pretrained`.
4. Run the script/notebook to launch the Gradio interface.

## Usage

1. Launch the app:
   ```
   python app.py
   ```
   Or run the notebook cells in Jupyter/Colab.

2. In the interface:
   - Upload an image (e.g., JPEG, PNG).
   - Enter a prompt in the "Query Input" textbox.
   - Adjust advanced settings if needed (e.g., increase max new tokens for longer responses).
   - Click "Process Image" to generate the output.
   - View results in the "Extracted Content" tab.
   - Switch to "Markdown Preview" for formatted viewing.
   - Use "Generate PDF & Render" to create a PDF with the image and text, customizable via PDF settings.
   - Download the PDF or view page previews in the gallery.

3. For best results:
   - Use high-quality images.
   - Craft detailed prompts for complex vision-language tasks.
   - Monitor GPU usage for large inputs.

## Model Details

- Model ID: `OpenGVLab/InternVL3_5-2B-Pretrained`
- Preprocessing: Dynamic image splitting into patches (up to 12) with normalization.
- Inference: Supports sampling-based generation with configurable parameters.
- Device: Automatically uses CUDA if available; falls back to CPU.

## Limitations

- The model does not support real-time streaming; full responses are generated at once.
- PDF generation requires valid image and text input.
- No internet access in the code interpreter environment; all dependencies must be pre-installed.
- Performance may vary based on hardware; GPU recommended for efficiency.

## Credits

- Model developed by OpenGVLab.
- Notebook by [prithivMLmods](https://huggingface.co/prithivMLmods) on Hugging Face.
- Built with Gradio for the UI, Transformers for model handling, and ReportLab/PyMuPDF for PDF features.

For issues or contributions, refer to the Hugging Face repository or open a pull request.
