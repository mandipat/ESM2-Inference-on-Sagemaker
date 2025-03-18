# esm
bash commands for converting into V2 version from OCI
aws ecr get-login-password --region us-east-1 | skopeo login --username AWS --password-stdin 576245601309.dkr.ecr.us-east-1.amazonaws.com

skopeo copy docker-daemon:esm2-sagemaker:v2 docker://576245601309.dkr.ecr.us-east-1.amazonaws.com/esm2-sagemaker:v1

