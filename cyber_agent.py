
import logging
import time
import torch
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Label mappings
LABEL_MAPPING = {
    "LABEL_0": "No Harassment",
    "LABEL_1": "Harassment",
    "LABEL_2": "Neutral",
    "LABEL_3": "Threat",
    "LABEL_4": "Insult"
}
FALLBACK_LABEL_MAPPING = {
    "POSITIVE": "No Harassment",
    "NEGATIVE": "Harassment"
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device set to use {device}")

# Load model and tokenizer
model_name = "D:/mldl-cp/model/mistral"
fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
is_fallback = False

try:
    if os.path.exists(model_name):
        logging.info(f"Model directory exists: {model_name}")
        logging.info(f"Directory contents: {os.listdir(model_name)}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        logging.info("Mistral model and tokenizer loaded successfully")
    else:
        logging.warning(f"Local model directory not found: {model_name}. Using fallback: {fallback_model}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
        is_fallback = True
        logging.info("Fallback DistilBERT model loaded")
    
    cyber_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=device if device.type == "cuda" else -1
    )
except Exception as e:
    logging.error(f"Failed to load Mistral model: {str(e)}. Falling back to {fallback_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
        is_fallback = True
        cyber_classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            device=device if device.type == "cuda" else -1
        )
        logging.info("Fallback DistilBERT model loaded after error")
    except Exception as fallback_e:
        logging.error(f"Failed to load fallback model: {str(fallback_e)}")
        raise RuntimeError("Unable to load any model")

def handle_cyber_agent(text: str, use_shap: bool =True, max_length: int = 128) -> dict:
    logging.info(f"Processing cyber agent query: {text[:100]}... (truncated)")
    start_time = time.time()
    
    try:
        # Truncate text
        tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        truncated_text = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
        logging.info(f"Truncated text for processing: {truncated_text[:100]}...")
        
        # Get predictions
        predictions = cyber_classifier(truncated_text)
        logging.info(f"Raw predictions: {predictions}")
        
        # Handle nested predictions
        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            predictions = predictions[0]
        
        # Process predictions
        label_scores = {}
        label_map = FALLBACK_LABEL_MAPPING if is_fallback else LABEL_MAPPING
        for pred in predictions:
            if isinstance(pred, dict) and "label" in pred and "score" in pred:
                label = label_map.get(pred["label"], pred["label"])
                label_scores[label] = float(pred["score"])
            else:
                logging.warning(f"Unexpected prediction format: {pred}")
                continue
        
        if not label_scores:
            raise ValueError("No valid predictions found")
        
        # Find highest-probability label
        max_label = max(label_scores, key=label_scores.get)
        max_score = label_scores[max_label]
        
        # Format scores as percentages
        formatted_scores = {k: f"{v * 100:.2f}%" for k, v in label_scores.items()}
        
        # Initialize result
        result = {
            "prediction": max_label,
            "probability": max_score,
            "prediction_scores": formatted_scores
        }
        
        # SHAP explanation
        if use_shap:
            try:
                explainer = SequenceClassificationExplainer(model, tokenizer)
                word_attributions = explainer(truncated_text)
                top_word = max(word_attributions, key=lambda x: abs(x[1]))[0]
                result["explanation"] = f"Most influential word: {top_word}"
            except Exception as e:
                logging.warning(f"SHAP explanation failed: {str(e)}")
                result["explanation"] = f"SHAP failed: {str(e)}"
        else:
            result["explanation"] = "SHAP skipped for faster processing"
        
        elapsed_time = time.time() - start_time
        logging.info(f"Cyber agent result: {result}, took {elapsed_time:.2f} seconds")
        return result
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Cyber agent error: {str(e)}, took {elapsed_time:.2f} seconds")
        return {"error": f"Cyber agent failed: {str(e)}"}