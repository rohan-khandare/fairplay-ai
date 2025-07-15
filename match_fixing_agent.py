
import pickle
import numpy as np
import shap
import logging

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and scaler
try:
    with open("models/match_fixing_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/match_fixing_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    logging.error(f"Failed to load match-fixing agent model/scaler: {str(e)}")
    raise

# Feature order for model
FEATURES = [
    "team_a_win_ratio", "team_b_win_ratio", "team_a_avg_goals", "team_b_avg_goals",
    "team_a_possession", "team_b_possession", "pre_match_odds_team_a", "pre_match_odds_team_b",
    "betting_volume_total", "betting_volume_on_draw", "unexpected_goals_timing",
    "red_cards", "social_sentiment_score"
]

def handle_match_fixing_agent(fields: dict) -> dict:
    try:
        # Ensure all features are present
        input_data = []
        for feature in FEATURES:
            if feature not in fields:
                logging.warning(f"Missing {feature} in match_fixing_agent input, already imputed")
                return {"error": f"Missing {feature} after imputation"}
            input_data.append(fields[feature])
        
        # Convert to numpy array and validate
        input_array = np.array([input_data], dtype=np.float64)
        logging.info(f"Match-fixing agent input features: {input_data}")
        
        if np.any(np.isnan(input_array)) or np.any(np.isinf(input_array)):
            logging.error(f"Invalid input data: contains NaN or Inf: {input_array}")
            return {"error": "Invalid input data: contains NaN or Inf"}
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        logging.info(f"Match-fixing agent scaled input: {input_scaled}")
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)
        logging.info(f"Match-fixing agent predict_proba output: {proba}, type: {type(proba)}")
        
        # Extract probability
        probability = proba[0][1]  # Probability of fixing (0 to 1)
        logging.info(f"Match-fixing agent raw probability: {probability}, type: {type(probability)}")
        
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
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        feature_importance = dict(zip(FEATURES, shap_values[0]))
        top_feature = max(feature_importance, key=lambda k: abs(feature_importance[k]))
        
        # Format probability safely
        probability_percent = probability * 100
        probability_str = f"{probability_percent:.2f}%"
        logging.info(f"Match-fixing agent formatted probability: {probability_str}")
        
        return {
            "prediction": "Fixed" if prediction == 1 else "Clean",
            "probability": probability_str,
            "explanation": f"Most influential factor: {top_feature} ({feature_importance[top_feature]:.4f})"
        }
    except Exception as e:
        logging.error(f"Match-fixing agent error: {str(e)}")
        return {"error": str(e)}