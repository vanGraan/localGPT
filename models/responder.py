# models/responder.py

from models.model_loader import load_model
from logger import get_logger
from PIL import Image
import os

logger = get_logger(__name__)

def generate_response(images, query, session_id, resized_height=280, resized_width=280, model_choice='qwen'):
    """
    Generates a response using the selected model based on the query and images.
    """
    try:
        logger.info(f"Generating response using model '{model_choice}'.")
        if model_choice == 'qwen':
            from qwen_vl_utils import process_vision_info
            # Load cached model
            model, processor, device = load_model('qwen')
            # Ensure dimensions are multiples of 28
            resized_height = (resized_height // 28) * 28
            resized_width = (resized_width // 28) * 28

            image_contents = []
            for image in images:
                image_contents.append({
                    "type": "image",
                    "image": os.path.join('static', image),
                    "resized_height": resized_height,
                    "resized_width": resized_width
                })
            messages = [
                {
                    "role": "user",
                    "content": image_contents + [{"type": "text", "text": query}],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            logger.info("Response generated using Qwen model.")
            return output_text[0]
        elif model_choice == 'gemini':
            from models.gemini_responder import generate_gemini_response
            model, processor = load_model('gemini')
            response = generate_gemini_response(images, query, model, processor)
            logger.info("Response generated using Gemini model.")
            return response
        elif model_choice == 'gpt4':
            from models.gpt4_responder import generate_gpt4_response
            model, _ = load_model('gpt4')
            response = generate_gpt4_response(images, query, model)
            logger.info("Response generated using GPT-4 model.")
            return response
        
        elif model_choice == 'llama-vision':
            # Load model, processor, and device
            model, processor, device = load_model('llama-vision')

            # Process images
            image_paths = [os.path.join('static', image) for image in images]
            # For simplicity, use the first image
            image_path = image_paths[0]
            image = Image.open(image_path).convert('RGB')

            # Prepare messages
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]}
            ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, return_tensors="pt").to(device)

            # Generate response
            output = model.generate(**inputs, max_new_tokens=512)
            response = processor.decode(output[0], skip_special_tokens=True)
            return response

        else:
                logger.error(f"Invalid model choice: {model_choice}")
                return "Invalid model selected."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "An error occurred while generating the response."
