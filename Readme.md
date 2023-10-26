#LLaVA: Visual Question Answering with Generative AI

LLaVA is an innovative Visual Question Answering (VQA) system powered by generative AI models. This repository allows you to effortlessly integrate state-of-the-art AI capabilities into your applications, enabling users to ask questions about images and receive accurate and human-like answers. LLaVA leverages advanced deep learning techniques to provide intelligent responses to visual queries.
Installation

Ensure you have Python installed on your system. Install the required dependencies using the following command:

bash

pip install -r requirements.txt

Usage

    Load Model Controllers and Workers:

    Before making any inferences, load the model controllers and workers by calling the load function from app.py. This function initializes the necessary components and listens on the defined ports specified in the environment file.

    python

from app import load

load()

Perform Model Inference:

To obtain answers to visual questions, call the model_inference function from app.py. Pass either a byte-encoded image or an image URL as input. The function takes the model name and question prompt from the environment variables and returns the corresponding answer.

python

    from app import model_inference

    image_bytes = ...  # Load your image as bytes or provide an image URL
    model_name = "LLaVA_Model"
    prompt = "What is happening in this image?"

    answer = model_inference(image_bytes, model_name, prompt)
    print("Answer:", answer)

    Logging (Optional):

    tools.py is included for logging purposes. You can use the logging functions provided in this file to monitor the application's behavior and performance.

Configuration

    Environment Variables:
        PORT: Port number for the server to listen on.
        MODEL_NAME: Name of the LLaVA model to use for question answering.
        PROMPT: Default question prompt for generating answers.

    Ensure these variables are correctly set in the .env file before running the application.

Example

python

from app import load, model_inference

# Load model controllers and workers
load()

# Perform model inference
image_bytes = ...  # Load your image as bytes or provide an image URL
model_name = "LLaVA_Model"
prompt = "What is happening in this image?"

answer = model_inference(image_bytes, model_name, prompt)
print("Answer:", answer)
