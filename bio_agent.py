
import pickle
import numpy as np
import shap
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and scaler
try:
    with open("models/xgboost_model_best.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    logging.error(f"Failed to load bio agent model/scaler: {str(e)}")
    raise

# Verify scaler type
if not isinstance(scaler, StandardScaler):
    logging.error(f"Invalid scaler type: {type(scaler)}. Expected StandardScaler.")
    raise TypeError(f"Invalid scaler type: {type(scaler)}")

# Feature order for model (19 features)
FEATURES = [
    "athlete_id", "sample_date", "hemoglobin", "rbc", "off_score", "t_e_ratio",
    "eaas_marker", "doping_detection_ratio", "steroid_mean", "steroid_min",
    "steroid_max", "age", "gender", "ph", "specific_gravity", "wbc", "rbc_urine",
    "protein", "glucose"
]

def handle_bio_agent(fields: dict) -> dict:
    try:
        # Ensure all features are present
        input_data = []
        for feature in FEATURES:
            if feature not in fields:
                if feature == "sample_date":
                    fields[feature] = "20240101"  # Impute internally
                    logging.info(f"Imputed {feature}: {fields[feature]}")
                else:
                    logging.warning(f"Missing {feature} in bio_agent input, already imputed")
                    return {"error": f"Missing {feature} after imputation"}
            value = fields[feature]
            # Convert sample_date to numeric (e.g., timestamp or dummy value)
            if feature == "sample_date":
                value = 20240101  # Dummy numeric value (e.g., YYYYMMDD)
            input_data.append(value)
        
        # Convert to numpy array and validate
        input_array = np.array([input_data], dtype=np.float64)
        logging.info(f"Bio agent input features: {input_data}")
        
        if np.any(np.isnan(input_array)) or np.any(np.isinf(input_array)):
            logging.error(f"Invalid input data: contains NaN or Inf: {input_array}")
            return {"error": "Invalid input data: contains NaN or Inf"}
        
        # Scale input
        try:
            input_scaled = scaler.transform(input_array)
            logging.info(f"Bio agent scaled input: {dict(zip(FEATURES, input_scaled[0]))}")
        except Exception as e:
            logging.error(f"Scaler transform error: {str(e)}")
            return {"error": f"Scaler transform error: {str(e)}"}
        
        # Predict
        try:
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)
            logging.info(f"Bio agent predict_proba output: {proba}, type: {type(proba)}")
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}
        
        # Extract probability
        probability = proba[0][1]  # Probability of doping (0 to 1)
        logging.info(f"Bio agent raw probability: {probability}, type: {type(probability)}")
        
        # Validate probability
        if not isinstance(probability, (int, float, np.floating)):
            logging.error(f"Invalid probability type: {type(probability)}, value: {probability}")
            return {"error": f"Invalid probability type: {type(probability)}"}
        
        # Convert to Python float and check for NaN/inf
        try:
            probability = float(probability)
            if np.isnan(probability) or np.isinf(probability):
                logging.error(f"Invalid probability value: {probability}")
                probability = 0.5  # Fallback
        except (TypeError, ValueError) as e:
            logging.error(f"Probability conversion error: {str(e)}, value: {probability}")
            probability = 0.5  # Fallback
        
        if not 0 <= probability <= 1:
            logging.warning(f"Probability out of range: {probability}, setting to 0.5")
            probability = 0.5
        
        # SHAP explanation
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)
            feature_importance = dict(zip(FEATURES, shap_values[0]))
            # Get top 3 features by absolute SHAP value
            top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            explanation = "Doping probability influenced by: " + "; ".join(
                # [f"{feat} (SHAP: {val:.4f}, Value: {fields[feat]})" for feat, val in top_features]
                [f"{feat} (SHAP: {val:.4f})" for feat, val in top_features]

            )
            logging.info(f"Bio agent SHAP values: {feature_importance}")
        except Exception as e:
            logging.error(f"SHAP error: {str(e)}")
            return {"error": f"SHAP error: {str(e)}"}
        
        # Format probability
        probability_percent = probability * 100
        probability_str = f"{probability_percent:.2f}%"
        logging.info(f"Bio agent formatted probability: {probability_str}")
        
        return {
            "prediction": "Doping" if prediction == 1 else "Clean",
            "probability": probability,  # Raw probability (0 to 1)
            "probability_percent": probability_str,  # Formatted percentage
            "explanation": explanation,
            "prediction_scores": {
                "Doping": probability_str,
                "Clean": f"{(1 - probability) * 100:.2f}%"
            }
        }
    except Exception as e:
        logging.error(f"Bio agent error: {str(e)}")
        return {"error": str(e)}