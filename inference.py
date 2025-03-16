import os
import json
import subprocess
# Install missing packages (only on first load)
def install_dependencies():
    try:
        import esm
    except ImportError:
        subprocess.check_call(["pip", "install", "fair-esm","torch","flask"])
import torch
import esm
from flask import Flask, request, jsonify


def model_fn(model_dir):
    # Load model
    model_path = os.path.join(model_dir, 'esm2_t12_35M_UR50D.pt')
    model,alphabet =esm.pretrained.load_model_and_alphabet_local(model_path)
    #model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()

    return {"model": model, "alphabet": alphabet}


app = Flask(__name__)

# Load model at startup
MODEL_DIR = "/opt/ml/model"
#MODEL_PATH = os.path.join(MODEL_DIR, "esm2_t12_35M_UR50D.pt")

# Load model and alphabet
print("Loading model...")
#
# model, alphabet = esm.pretrained.load_model_and_alphabet_local(MODEL_PATH)
#model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
#model.eval()
#batch_converter = alphabet.get_batch_converter()
#print("Model loaded successfully.")

@app.route('/ping', methods=['GET'])
def ping():
    """Health check - SageMaker requires this endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/invocations', methods=['POST'])
def inference():
    """Handle inference requests from SageMaker"""
    try:
        content_type = request.content_type
        if content_type != "application/json":
            return jsonify({"error": "Unsupported content type: {}".format(content_type)}), 400

        # Parse input data
        input_data = request.get_json()
        if "masked_sequence" not in input_data:
            return jsonify({"error": "Missing 'masked_sequence' in input data."}), 400

        sequence = input_data["masked_sequence"]
        model= model_fn(MODEL_DIR)["model"]
        alphabet = model_fn(MODEL_DIR)["alphabet"]
        batch_converter = alphabet.get_batch_converter()
        # Convert sequence into tokens
        data = [("sequence", sequence)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(next(model.parameters()).device)

        # Perform inference
        with torch.no_grad():
            outputs = model(tokens)

        # Extract logits
        logits = outputs["logits"]

        # Generate unmasked sequence (argmax over logits)
        predicted_tokens = logits.argmax(dim=-1)
        predicted_token_ids = predicted_tokens[0].cpu().numpy().tolist()
        generated_sequence = "".join([alphabet.get_tok(token_id) for token_id in predicted_token_ids])

        # Extract embedding vector (e.g., first token embedding)
        embedding_vector = logits[0, 0, :].tolist()

        response = {
            "generated_sequence": generated_sequence,
            "embedding": embedding_vector
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
