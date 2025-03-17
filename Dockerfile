# Use PyTorch base image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create the directory for model artifacts.
RUN mkdir -p /opt/ml/model

# Copy your local model weights into the model directory.
COPY esm2_t12_35M_UR50D.pt /opt/ml/model/esm2_t12_35M_UR50D.pt
COPY esm2_t12_35M_UR50D-contact-regression.pt /opt/ml/model/esm2_t12_35M_UR50D-contact-regression.pt


# Set the working directory.
WORKDIR /app

# Copy your application code.
COPY inference.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh

# Expose the port that the Flask app runs on.
EXPOSE 8080

# Set the entrypoint.
ENTRYPOINT ["/app/entrypoint.sh"]

# Set the default command to "serve".
CMD ["serve"]

