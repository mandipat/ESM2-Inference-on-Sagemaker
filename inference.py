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

def model_fn(MODEL_PATH):
    """
    Load the ESM2 model and its alphabet using the esm.pretrained API.
    This method downloads the model if needed and sets it to evaluation mode.
    """
    try:
        model_path = os.environ.get(MODEL_PATH, "/opt/ml/model/")
        model_location = os.path.join(model_path, "esm2_t12_35M_UR50D.pt")
        if not os.path.exists(model_location):
            logger.info("Downloading ESM2 model from AWS S3")
            model,alphabet= esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
            logger.info("ESM2 model downloaded successfully")
        else:
            logger.info("Model already exists at %s", model_location)
            logger.info("Loading ESM2 model using esm.pretrained")
            try:
                import argparse
                import torch.serialization
                torch.serialization.add_safe_globals([argparse.Namespace])
                model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location)
                logger.info("ESM2 modead with local model successfully")
            except Exception as e:
                logger.error("Error loading model: %s", e)
                try:
                    logging.info(" Trying fallback model: esm2_t6_8M_UR50D")
                    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                    model.eval()
                    logging.info(" Fallback model downloaded successfully!")
                except Exception as e:
                    logging.error("Error loading fallback model: %s", e)
                    raise e

        model.eval()
        logger.info("ESM2 model loaded successfully")
        return model, alphabet
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise e
        

# Load the model and alphabet on startup.
model_alphabet = model_fn("/opt/ml/model/")

def input_fn(request_body, request_content_type='application/json'):
    """
    Process the input JSON data.
    Expected JSON format:
      {
         "sequence": "YOUR_PROTEIN_SEQUENCE",
         "task": "embedding"  // or "get_shape" or "fill_mask"
      }
    If "task" is not provided, defaults to "embedding".
    """
    if request_content_type == 'application/json':
        try:
            data = json.loads(request_body)
            sequence = data.get('sequence', '')
            if not sequence:
                raise ValueError("No sequence provided")
            task = data.get('task', 'embedding')
            logger.info("Received sequence (first 10 chars): %s", sequence[:10])
            logger.info("Task: %s", task)
            return {"sequence": sequence, "task": task}
        except Exception as e:
            logger.error("Error processing input: %s", e)
            raise e
    else:
        raise ValueError("Unsupported content type: " + request_content_type)

def predict_fn(inputs, model_alphabet):
    """
    Depending on the requested task, run one of the following:
      - "embedding": Return the averaged embedding from layer 12.
      - "get_shape": Return the shape of the token representations from layer 12.
      - "fill_mask": Fill a masked token in the sequence.
    
    For "fill_mask", the sequence must include a mask token.
    The alphabet (esm.Alphabet) is used for tokenization and for retrieving the mask index.
    """
    sequence = inputs["sequence"]
    task = inputs["task"]
    model, alphabet = model_alphabet

    # Tokenize the input sequence.
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if task == "embedding":
        # Get token representations and average them (excluding special tokens).
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12], return_contacts=False)
        token_representations = results["representations"][12]
        embedding = token_representations[0, 1:-1].mean(dim=0)
        logger.info("Embedding shape: %s", embedding.shape)
        return {"embedding": embedding.cpu().numpy().tolist()}

    elif task == "get_shape":
        # Return the shape of the token representations from layer 12.
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12], return_contacts=False)
        token_representations = results["representations"][12]
        shape = list(token_representations.shape)
        logger.info("Token representations shape: %s", shape)
        return {"shape": shape}

    elif task == "fill_mask":
        # Get the mask token id from the alphabet.
        mask_idx = alphabet.mask_idx
        # Clone the token tensor to modify it.
        tokens = batch_tokens[0].clone()
        # Identify all positions where the token equals the mask token id.
        masked_positions = (tokens == mask_idx).nonzero(as_tuple=True)[0]
        if len(masked_positions) == 0:
            raise ValueError("No mask token found in the sequence")
        # Run inference to obtain logits.
        with torch.no_grad():
            output = model(batch_tokens, repr_layers=[12], return_contacts=False)
        if "logits" not in output:
            raise ValueError("Model does not provide logits for fill_mask task")
        logits = output["logits"]
        # For each masked position, get the predicted token id.
        predicted_token_ids = logits[0, masked_positions, :].argmax(dim=-1)
        # Replace the mask token ids with the predicted token ids.
        tokens[masked_positions] = predicted_token_ids
        # Remove the special tokens (assume first and last tokens are special).
        token_ids = tokens[1:-1].tolist()
        # Convert token ids to their string representations.
        tokens_str = [alphabet.get_tok(i) for i in token_ids]
        # Combine tokens into a filled sequence.
        filled_sequence = "".join(tokens_str)
        logger.info("Filled sequence: %s", filled_sequence)
        return {"filled_sequence": filled_sequence}

    else:
        raise ValueError(f"Unsupported task: {task}")

def output_fn(prediction, accept='application/json'):
    """
    Format the prediction output as a JSON string.
    """
    return json.dumps(prediction), accept


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
    # Running the Flask server on port 8080 
    app.run(host='0.0.0.0', port=8080)
