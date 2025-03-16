import os
import json
import logging
from flask import Flask, request, jsonify
import torch
import esm

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Flask(__name__)

def model_fn():
    """
    Load the ESM2 model and its alphabet using the esm.pretrained API.
    This method downloads the model if needed and sets it to evaluation mode.
    """
    try:
        logger.info("Loading ESM2 model using esm.pretrained")
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        model.eval()
        logger.info("ESM2 model loaded successfully")
        return model, alphabet
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise e

# Load the model and alphabet on startup.
model_alphabet = model_fn()

def input_fn(request_body, request_content_type='application/json'):
    """
    Process the input JSON data. Expected format:
    {"sequence": "YOUR_PROTEIN_SEQUENCE"}
    """
    if request_content_type == 'application/json':
        try:
            data = json.loads(request_body)
            sequence = data.get('sequence', '')
            if not sequence:
                raise ValueError("No sequence provided")
            logger.info("Received sequence (first 10 chars): %s", sequence[:10])
            return sequence
        except Exception as e:
            logger.error("Error processing input: %s", e)
            raise e
    else:
        raise ValueError("Unsupported content type: " + request_content_type)

def predict_fn(sequence, model_alphabet):
    """
    Tokenize the sequence using the alphabet's batch converter,
    run inference to obtain token representations from layer 12,
    and compute a fixed-length embedding by averaging (excluding special tokens).
    """
    model, alphabet = model_alphabet
    try:
        batch_converter = alphabet.get_batch_converter()
        # Create a single-item batch; the first element is an arbitrary identifier.
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        logger.info("Converted sequence to tokens")
    except Exception as e:
        logger.error("Error during tokenization: %s", e)
        raise e

    try:
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12], return_contacts=False)
            logger.info("Model inference completed")
    except Exception as e:
        logger.error("Error during model inference: %s", e)
        raise e

    try:
        token_representations = results["representations"][12]
        # Exclude start and end tokens and average the token representations
        sequence_representation = token_representations[0, 1:-1].mean(dim=0)
        logger.info("Obtained sequence representation of shape: %s", sequence_representation.shape)
    except Exception as e:
        logger.error("Error processing model output: %s", e)
        raise e

    return sequence_representation.cpu().numpy()

def output_fn(prediction, accept='application/json'):
    """
    Format the prediction (embedding) as a JSON string.
    """
    response = {'embedding': prediction.tolist()}
    return json.dumps(response), accept

@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint to verify if the model is loaded.
    """
    health = model_alphabet is not None
    status = 200 if health else 404
    return jsonify({'status': 'Healthy' if health else 'Unhealthy'}), status

@app.route('/invocations', methods=['POST'])
def invocations():
    """
    Process incoming requests: parse input, run model inference,
    and return the embedding in JSON format.
    """
    data = request.data.decode('utf-8')
    content_type = request.content_type
    try:
        # Process the input sequence
        sequence = input_fn(data, content_type)
        # Run inference
        prediction = predict_fn(sequence, model_alphabet)
        # Format the output
        response, out_content_type = output_fn(prediction, content_type)
        logger.info("Request processed successfully")
        return response, 200, {'Content-Type': out_content_type}
    except Exception as e:
        logger.error("Error in /invocations: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask server on port 8080 (SageMaker expects this port)
    app.run(host='0.0.0.0', port=8080)
