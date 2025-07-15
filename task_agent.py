
import logging
import re
from typing import List, Dict
from bio_agent import handle_bio_agent
from match_fixing_agent import handle_match_fixing_agent
from cyber_agent import handle_cyber_agent

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s - %(levelname)s - %(message)s")

AGENTS = {
    "doping": handle_bio_agent,
    "match-fixing": handle_match_fixing_agent,
    "cyber violation": handle_cyber_agent
}

# Default values for imputation (only for missing fields)
DEFAULT_VALUES = {
    "off_score": 90,
    "steroid_mean": 5,
    "steroid_min": 2,
    "steroid_max": 8,
    "ph": 6.0,
    "specific_gravity": 1.015,
    "wbc": 7,
    "rbc_urine": 0,
    "protein": 0,
    "glucose": 0,
    "sample_date": "20240101",  # Added default for sample_date
    "team_a_avg_goals": 1.5,
    "team_b_avg_goals": 1.5,
    "team_a_possession": 0.5,
    "team_b_possession": 0.5,
    "pre_match_odds_team_a": 2.0,
    "pre_match_odds_team_b": 2.0,
    "betting_volume_on_draw": 2000
}

def split_query(query: str) -> List[str]:
    """
    Split query into sub-queries based on sentence-ending periods, avoiding splits within numbers.
    """
    sub_queries = []
    current_query = ""
    i = 0
    while i < len(query):
        current_query += query[i]
        if query[i] == ".":
            # Check if the period is part of a decimal number
            if i > 0 and i < len(query) - 1:
                prev_char = query[i-1]
                next_char = query[i+1]
                # If surrounded by digits (e.g., 17.5), don't split
                if prev_char.isdigit() and next_char.isdigit():
                    i += 1
                    continue
            # Split on sentence-ending periods (followed by space or end)
            if (i == len(query) - 1 or query[i+1].isspace()) and len(current_query.strip()) > 10:
                sub_queries.append(current_query.strip())
                current_query = ""
        i += 1
    if current_query.strip():
        sub_queries.append(current_query.strip())
    logging.info(f"Split sub-queries: {sub_queries}")
    return sub_queries

def classify_intent(sub_query: str) -> List[Dict]:
    intents = []
    sub_query_lower = sub_query.lower()
    
    if any(k in sub_query_lower for k in ["hemoglobin", "rbc", "testosterone", "eaas_marker"]):
        intents.append({"intent": "doping", "confidence": 0.9})
    if any(k in sub_query_lower for k in ["team_a_win_ratio", "betting_volume", "red_cards"]):
        intents.append({"intent": "match-fixing", "confidence": 0.8})
    if any(k in sub_query_lower for k in ["instagram", "social media", "harassment"]):
        intents.append({"intent": "cyber violation", "confidence": 0.7})
    
    logging.info(f"Classified intents for sub_query '{sub_query[:50]}...': {intents}")
    return intents if intents else [{"intent": "unknown", "confidence": 0.1}]

def extract_fields(sub_query: str, intent: str) -> tuple:
    fields = {}
    is_valid = True
    imputed_fields = []
    missing_fields = []
    
    if intent == "doping":
        required_fields = ["hemoglobin", "rbc", "t_e_ratio", "eaas_marker", "doping_detection_ratio", "age", "gender"]
        patterns = {
            "athlete_id": r"athlete_id\s*=\s*(\d+)",
            "hemoglobin": r"hemoglobin\s*=\s*([-]?\d*\.?\d*)",
            "rbc": r"rbc\s*=\s*([-]?\d*\.?\d*)",
            "t_e_ratio": r"t_e_ratio\s*=\s*([-]?\d*\.?\d*)",
            "eaas_marker": r"eaas_marker\s*=\s*([-]?\d*\.?\d*)",
            "doping_detection_ratio": r"doping_detection_ratio\s*=\s*([-]?\d*\.?\d*)",
            "off_score": r"off_score\s*=\s*([-]?\d*\.?\d*)",
            "steroid_mean": r"steroid_mean\s*=\s*([-]?\d*\.?\d*)",
            "steroid_min": r"steroid_min\s*=\s*([-]?\d*\.?\d*)",
            "steroid_max": r"steroid_max\s*=\s*([-]?\d*\.?\d*)",
            "sample_date": r"sample_date\s*=\s*([\d-]+)",
            "age": r"age\s*=\s*(\d+)",
            "gender": r"gender\s*=\s*(\d)",
            "ph": r"ph\s*=\s*([-]?\d*\.?\d*)",
            "specific_gravity": r"specific_gravity\s*=\s*([-]?\d*\.?\d*)",
            "wbc": r"wbc\s*=\s*([-]?\d*\.?\d*)",
            "rbc_urine": r"rbc_urine\s*=\s*([-]?\d*\.?\d*)",
            "protein": r"protein\s*=\s*([-]?\d*\.?\d*)",
            "glucose": r"glucose\s*=\s*([-]?\d*\.?\d*)"
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, sub_query, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    fields[field] = float(value) if field != "sample_date" else value
                    logging.info(f"Extracted {field}: {value}")
                except ValueError:
                    logging.warning(f"Invalid value for {field}: {value}")
                    if field in required_fields:
                        is_valid = False
                        missing_fields.append(field)
            elif field in DEFAULT_VALUES:
                fields[field] = DEFAULT_VALUES[field]
                imputed_fields.append(field)
                logging.info(f"Imputed {field}: {fields[field]}")
            elif field in required_fields:
                is_valid = False
                missing_fields.append(field)
                logging.warning(f"Missing required field: {field}")
        
        # Ensure all required fields are present
        for field in required_fields:
            if field not in fields:
                is_valid = False
                missing_fields.append(field)
                logging.warning(f"Required field {field} not found in extracted fields")
        
        # Log all fields
        logging.info(f"Extracted doping fields: {fields}")
    
    elif intent == "match-fixing":
        required_fields = ["team_a_win_ratio", "team_b_win_ratio", "red_cards", "betting_volume_total", "unexpected_goals_timing", "social_sentiment_score"]
        patterns = {
            "team_a_win_ratio": r"team_a_win_ratio\s*=\s*([-]?\d*\.?\d*)",
            "team_b_win_ratio": r"team_b_win_ratio\s*=\s*([-]?\d*\.?\d*)",
            "red_cards": r"red_cards\s*=\s*(\d+)",
            "betting_volume_total": r"betting_volume_total\s*=\s*([-]?\d*\.?\d*)",
            "unexpected_goals_timing": r"unexpected_goals_timing\s*=\s*(\d+)",
            "social_sentiment_score": r"social_sentiment_score\s*=\s*([-]?\d*\.?\d*)",
            "team_a_avg_goals": r"team_a_avg_goals\s*=\s*([-]?\d*\.?\d*)",
            "team_b_avg_goals": r"team_b_avg_goals\s*=\s*([-]?\d*\.?\d*)",
            "team_a_possession": r"team_a_possession\s*=\s*([-]?\d*\.?\d*)",
            "team_b_possession": r"team_b_possession\s*=\s*([-]?\d*\.?\d*)",
            "pre_match_odds_team_a": r"pre_match_odds_team_a\s*=\s*([-]?\d*\.?\d*)",
            "pre_match_odds_team_b": r"pre_match_odds_team_b\s*=\s*([-]?\d*\.?\d*)",
            "betting_volume_on_draw": r"betting_volume_on_draw\s*=\s*([-]?\d*\.?\d*)"
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, sub_query, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    fields[field] = float(value)
                    logging.info(f"Extracted {field}: {value}")
                except ValueError:
                    logging.warning(f"Invalid value for {field}: {value}")
                    if field in required_fields:
                        is_valid = False
                        missing_fields.append(field)
            elif field in DEFAULT_VALUES:
                fields[field] = DEFAULT_VALUES[field]
                imputed_fields.append(field)
                logging.info(f"Imputed {field}: {fields[field]}")
            elif field in required_fields:
                is_valid = False
                missing_fields.append(field)
                logging.warning(f"Missing required field: {field}")
        
        # Ensure all required fields are present
        for field in required_fields:
            if field not in fields:
                is_valid = False
                missing_fields.append(field)
                logging.warning(f"Required field {field} not found in extracted fields")
        
        # Log all fields
        logging.info(f"Extracted match-fixing fields: {fields}")
    
    return fields, is_valid, imputed_fields, missing_fields

def handle_query(query: str) -> List[Dict]:
    logging.info(f"Starting handle_query with query: {query}")
    results = []
    sub_queries = split_query(query)
    
    for sub_query in sub_queries:
        intents = classify_intent(sub_query)
        for intent_dict in intents:
            intent = intent_dict["intent"]
            confidence = intent_dict["confidence"]
            logging.info(f"Processing intent: {intent}, confidence: {confidence}")
            if confidence < 0.5 or intent not in AGENTS:
                logging.warning(f"Skipping intent {intent} (confidence {confidence} or invalid agent)")
                continue
            
            fields, is_valid, imputed_fields, missing_fields = extract_fields(sub_query, intent)
            if not is_valid and intent != "cyber violation":
                result = {
                    "sub_query": sub_query,
                    "intent": intent,
                    "confidence": confidence,
                    "error": f"Missing required fields: {missing_fields}",
                    "imputed_fields": imputed_fields
                }
                results.append(result)
                logging.error(f"Invalid fields for {intent}: {result['error']}")
                continue
            
            try:
                if intent == "cyber violation":
                    agent_result = AGENTS[intent](sub_query, use_shap=True)
                else:
                    agent_result = AGENTS[intent](fields)
                logging.info(f"Agent result for {intent}: {agent_result}")
                
                # Standardize output
                if intent == "cyber violation":
                    if "error" in agent_result:
                        standardized_result = agent_result
                    else:
                        standardized_result = {
                            "prediction": agent_result.get("prediction", "N/A"),
                            "probability": agent_result.get("probability", 0.0),
                            "explanation": agent_result.get("explanation", "No explanation"),
                            "prediction_scores": agent_result.get("prediction_scores", {})
                        }
                else:
                    standardized_result = agent_result
                
                results.append({
                    "sub_query": sub_query,
                    "intent": intent,
                    "confidence": confidence,
                    "result": standardized_result,
                    "imputed_fields": imputed_fields
                })
            except Exception as e:
                result = {
                    "sub_query": sub_query,
                    "intent": intent,
                    "confidence": confidence,
                    "error": f"Agent error: {str(e)}",
                    "imputed_fields": imputed_fields
                }
                results.append(result)
                logging.error(f"Agent error for {intent}: {str(e)}")
    
    logging.info(f"Final results: {results}")
    return results