# models/model_loader.py

import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from logger import get_logger

logger = get_logger(__name__)

# Cache for loaded models
_model_cache = {}

def detect_device():
    """
    Detects the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def load_model(model_choice):
    """
    Loads and caches the specified model.
    """
    global _model_cache

    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]

    if model_choice == 'qwen':
        device = detect_device()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model.to(device)
        _model_cache[model_choice] = (model, processor, device)
        logger.info("Qwen model loaded and cached.")
        return _model_cache[model_choice]
    elif model_choice == 'gemini':
        # Load Gemini model
        import genai
        genai.api_key = os.environ.get('GENAI_API_KEY')
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        processor = None
        _model_cache[model_choice] = (model, processor)
        logger.info("Gemini model loaded and cached.")
        return _model_cache[model_choice]
    elif model_choice == 'gpt4':
        # Load OpenAI GPT-4 model
        import openai
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        _model_cache[model_choice] = (None, None)
        logger.info("GPT-4 model ready and cached.")
        return _model_cache[model_choice]
    else:
        logger.error(f"Invalid model choice: {model_choice}")
        raise ValueError("Invalid model choice.")
