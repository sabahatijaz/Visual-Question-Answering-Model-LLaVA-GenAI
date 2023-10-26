# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
WORKDIR /app
RUN git clone https://github.com/haotian-liu/LLaVA.git

RUN apt-get install git-lfs

RUN git lfs install
RUN git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
WORKDIR /app/LLaVA
RUN python -m pip install --upgrade pip && apt-get update
RUN pip install -e .
RUN pip install flash-attn --no-build-isolation
RUN git pull
RUN pip uninstall transformers -y
RUN pip install -e .

RUN pip install -r requirements.txt
# Run your application using CMD which is the command that is executed when the container starts
CMD ["python", "app.py"]
