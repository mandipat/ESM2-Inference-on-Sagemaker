# Use PyTorch base image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install necessary Python packages
RUN pip install --upgrade pip

# Install fair-esm from GitHub
RUN pip install git+https://github.com/facebookresearch/esm.git

# Install SageMaker Inference Toolkit
RUN pip install sagemaker-inference 

# Set working directory
WORKDIR /opt/ml/code

RUN pip install flask torch

COPY inference.py /opt/ml/code/inference.py

# Copy both model files (main model & regression weights) to the correct directory
COPY esm2_t12_35M_UR50D.pt /opt/ml/model/esm2_t12_35M_UR50D.pt
COPY esm2_t12_35M_UR50D-contact-regression.pt /opt/ml/model/esm2_t12_35M_UR50D-contact-regression.pt

ENV SAGEMAKER_PROGRAM=inference.py \
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code \
    SAGEMAKER_CONTAINER_LOG_LEVEL=20 \
    SAGEMAKER_MULTI_MODEL=false \
    PATH="/opt/ml/code:${PATH}"
# Create serve.sh in your project directory

# # # Create serve script
# COPY test.py /opt/ml/code/test.py
# RUN echo '#!/bin/bash\n\
# python test.py' > /usr/local/bin/serve && \
# chmod +x /usr/local/bin/serve
EXPOSE 8080
# # Set the entry point

ENTRYPOINT ["python", "/opt/ml/code/inference.py"]


