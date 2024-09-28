# models/responder.py

from models.model_loader import load_model
from transformers import GenerationConfig
from logger import get_logger
from PIL import Image
import torch
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
        
        elif model_choice == "pixtral":

            model, sampling_params, device = load_model('pixtral')

            image_urls = []
            for img in images:
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                image_urls.append(f"data:image/png;base64,{img_str}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
                    ]
                },
            ]

            outputs = model.chat(messages, sampling_params=sampling_params)
            return outputs[0].outputs[0].text
        
        elif model_choice == "molmo":
            
            model, processor, device = load_model('molmo')
            pil_images = []
            for img_path in images:
                full_path = os.path.join('static', img_path)
                if os.path.exists(full_path):
                    try:
                        img = Image.open(full_path).convert('RGB')
                        pil_images.append(img)
                    except Exception as e:
                        logger.error(f"Error opening image {full_path}: {e}")
                else:
                    logger.warning(f"Image file not found: {full_path}")

            if not pil_images:
                return "No images could be loaded for analysis."

            try:
                # Log the types and shapes of the images
                logger.info(f"Number of images: {len(pil_images)}")
                logger.info(f"Image types: {[type(img) for img in pil_images]}")
                logger.info(f"Image sizes: {[img.size for img in pil_images]}")

                # Process the images and text
                inputs = processor.process(
                    images=pil_images,
                    text=query
                )

                # Log the keys and shapes of the inputs
                logger.info(f"Input keys: {inputs.keys()}")
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"Input '{k}' shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
                    else:
                        logger.info(f"Input '{k}' type: {type(v)}")

                # Move inputs to the correct device and make a batch of size 1
                inputs = {k: v.to(model.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Log the updated shapes after moving to device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"Updated input '{k}' shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")

                # Generate output
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )

                # Only get generated tokens; decode them to text
                generated_tokens = output[0, inputs['input_ids'].size(1):]
                generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            except Exception as e:
                logger.error(f"Error in Molmo processing: {str(e)}", exc_info=True)
                return f"An error occurred while processing the images: {str(e)}"
            finally:
                # Close the opened images to free up resources
                for img in pil_images:
                    img.close()

            return generated_text
        else:
            logger.error(f"Invalid model choice: {model_choice}")
            return "Invalid model selected."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "An error occurred while generating the response."
