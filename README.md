# ESM2 Model Deployment on AWS SageMaker

## Project Overview
This project demonstrates the deployment of Facebook's ESM2 (Evolutionary Scale Modeling) protein language model as a SageMaker endpoint for inference tasks. The implementation includes containerization, Flask API development, and AWS integration.

## Technical Stack
- **Model**: ESM2 (esm2_t12_35M_UR50D)
- **Framework**: PyTorch, Flask
- **Cloud**: AWS (ECR, SageMaker)
- **Container**: Docker
- **Languages**: Python, Bash

## Project Structure
```
ESM2/
├── Dockerfile              # Container configuration
├── entrypoint.sh          # Container entry point script
├── ESM.drawio            # Architecture diagram
├── inference.py          # Flask application for model serving
├── requirements.txt      # Python dependencies
├── esm2_t12_35M_UR50D.pt # Base model
├── esm2_t12_35M_UR50D-contact-regression.pt # Contact prediction model
```

## Setup and Deployment

### 1. Local Development
```bash
# Build Docker image
docker build -t esm2-sagemaker:v2 .

# Run container locally
docker run -p 8080:8080 esm2-sagemaker:v2 serve
```

### 2. AWS Deployment
```bash
# Login to AWS ECR
aws ecr get-login-password --region us-east-1 | skopeo login --username AWS --password-stdin 576245601309.dkr.ecr.us-east-1.amazonaws.com

# Push image to ECR
skopeo copy docker-daemon:esm2-sagemaker:v2 docker://576245601309.dkr.ecr.us-east-1.amazonaws.com/esm2-sagemaker:v1
```

## API Endpoints

### Health Check
- **Endpoint**: `/ping`
- **Method**: GET
- **Response**: 200 OK if model is loaded

### Inference
- **Endpoint**: `/invocations`
- **Method**: POST
- **Input Format**:
```json
{
    "sequence": "MGQLVFSVALLFCLVTQAS",
    "task": "embedding"  // or "get_shape" or "fill_mask"
}
```
- **Output**: Model predictions based on the specified task

## Example Usage

```python
import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('runtime.sagemaker')

# Sample protein sequence
payload = {
    "sequence": "MGQLVFSVALLFCLVTQAS",
    "task": "embedding"
}

# Send request to endpoint
response = runtime.invoke_endpoint(
    EndpointName='esm2-endpoint',
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse response
result = json.loads(response['Body'].read().decode())
```

## Model Details
- **Architecture**: ESM2-t12 (35M parameters)
- **Input**: Protein sequences in amino acid format
- **Output**: 
  - Embeddings (768-dimensional vectors)
  - Token predictions for masked sequences
  - Contact predictions (optional)

## Performance Considerations
- Maximum sequence length: 1024 tokens
- Batch processing available
- GPU acceleration supported
- Average inference time: ~100ms per sequence

## Monitoring and Maintenance
- CloudWatch metrics enabled for endpoint monitoring
- Auto-scaling configured based on request volume
- Regular model updates through container versioning

## Security
- IAM roles configured for minimal required permissions
- VPC endpoints for private network access
- Encrypted data transfer using AWS KMS

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

