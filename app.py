from tools import get_handler as get_logger
from llava.serve.cli import main
from pydantic import BaseModel
from decouple import config
from fastapi import FastAPI
from io import BytesIO
from PIL import Image
import subprocess
import threading
import requests
import argparse
import uvicorn
import logging
import base64
import time
import json
import os

logger = get_logger(log_file='LLaVA_model.log', log_level=logging.INFO)
logger.info("Model dependencies imported and logger configured")

app = FastAPI()


class InputData(BaseModel):
    """Input data schema for the FastAPI endpoint."""
    image_path: str


def load_image(image_file):
    try:
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image.save("input.jpg")
            return "input.jpg"
        else:
            image = base64.b64decode(image_file)
            image = Image.open(BytesIO(image)).convert('RGB')
            image.save("input.jpg")
            return "input.jpg"
    except Exception as err:
        print("error in load image: ", err)


def Run2(model_path, prompt, image):
    image_path = image
    message = prompt
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    headers = {"User-Agent": "LLaVA Client"}
    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n {message} ASSISTANT:"
    pload = {
        "model": model_path,
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_new_tokens": 1024,
        'stop': '</s>',
        "images": [str(encoded_image)],
    }
    output = ""
    try:
        # Stream output
        response = requests.post(f"http://localhost:{str(config("WORKER_PORT"))}" + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            # print(chunk)
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    # print(output)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                time.sleep(0.03)

    except requests.exceptions.RequestException as e:
        print("error: ", e)
    return output


def Run(model_path, prompt, image):
    """
        Run the LLAVA model with the provided input data.

        Args:
            model_path (str): Path to the model.
            prompt (str): Prompt for the model.
            image (str): Path to the image.

        Returns:
            str: Model output.
        """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=prompt)
    parser.add_argument("--image-file", type=str, default=image)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')

    # Parse known arguments and unknown arguments
    args, unknown_args = parser.parse_known_args()
    logger.info(args)
    output = main(args)
    return output


def run_llava_model_worker():
    """Start the LLAVA model worker as a subprocess."""
    controller_port = str(config("CONTROLLER_PORT"))
    worker_port = str(config("WORKER_PORT"))
    model_path = str(config("MODEL_PATH"))
    command = f"python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:{controller_port} --port {worker_port} --worker http://localhost:{worker_port} --model-path {model_path}"
    process = subprocess.Popen(command, shell=True)
    process.communicate()  # Wait for the process to finish


def run_llava_serve():
    """Start the LLAVA server as a subprocess."""
    controller_port =str(config("CONTROLLER_PORT"))
    command = f"python -m llava.serve.controller --host 0.0.0.0 --port {controller_port}"
    process = subprocess.Popen(command, shell=True)
    process.communicate()  # Wait for the process to finish


def load():
    """Load the LLAVA controller and workers."""
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    thread_serve = threading.Thread(target=run_llava_serve)
    thread_model_worker = threading.Thread(target=run_llava_model_worker)

    thread_serve.start()
    time.sleep(10)
    thread_model_worker.start()
    time.sleep(10)


def model_inference(image):
    model_path = config("MODEL_PATH")
    prompt = config("PROMPT")
    image_path = load_image(image)
    print(image_path)
    respo = Run2(model_path, prompt, image_path)
    return respo


@app.post("/run")
async def run_model(data: InputData):
    """
        FastAPI endpoint to run the LLAVA model with provided input data.

        Args:
            data (InputData): Input data containing model_path, prompt, and image.

        Returns:
            dict: Response message indicating the model is running.
        """
    print(data)
    model_path = "liuhaotian/llava-v1.5-13b"  # config("MODEL_PATH")
    prompt = """Describe in detail what this person is wearing, using the categories: (Top, Bottom, Outerwear, Accessories, Footwear, Outfit)(Outfit means a single piece of clothing that covers the body, such as a dress or overalls). Return a JSON with descriptions for each category, and nothing else. example:
    {Top: turtleneck, Bottom: Black trousers, Outerwear: Dark grey coat, Accessories: Black gloves, Footwear: Black leather oxfords, Outfit: None }
    { Top: White tank top, Bottom: Blue jeans, Outerwear: Gold cardigan with patterned sequins, Accessories: Gold earrings and gold bracelets, Footwear: Gold pointed toe heels, Outfit: None },
    {Top: None, Bottom: None, Outerwear: None, Accessories: Black sunglasses, Footwear: Beige high heel sandals, Outfit: Dress with horizontal zigzag stripes that are purple, blue, and green },
    The examples do not have quotes strings but please add them in your results. Describe that in even greater detail. For example, you can expand on the graphic on the t- shirt, and expand on what logo or shape is on the sneakers?"""  # config("PROMPT")
    image_path = load_image(data.image_path)
    thread = threading.Thread(target=Run, args=(model_path, prompt, image_path))
    thread.start()
    return {"message": "Model is running in the background."}


if __name__ == "__main__":
    load()
    logger.info("LLaVA controller and workers loaded")
    app_port = 7860  # int(config("MODEL_PORT"))
    uvicorn.run(app, host="0.0.0.0", port=app_port)
