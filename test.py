import inference as inference

def test_inference():
    print("Loading model...")
    model = inference.model_fn("/opt/ml/model")
    #print(model)
    
    # Test input
    test_input= "MGYARVNAKTDVA"
    print("Running inference...")
    result = inference.predict_fn(test_input, model)
    print("Inference result:", result)

if __name__ == "__main__":
    test_inference()
