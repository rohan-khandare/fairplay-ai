# 🏅 FairPlay AI

**FairPlay AI** is an intelligent, modular system for detecting sports violations such as **doping**, **match-fixing**, and **cyber/social media abuse**. It combines **Large Language Models (LLMs)**, **domain-specific AI agents**, and **Explainable AI (XAI)** techniques to provide accurate, interpretable decisions from natural language queries.

---

## 🚀 Features

- 🧠 **Multi-Agent System**: Separate AI agents for Doping, Match-Fixing, and Cyber Violation detection.
- 🤖 **LLM-Powered Query Parsing**: Natural language interface using Groq's free LLM (LLaMA3).
- 🧪 **Doping Detection**: XGBoost model using biomedical data.
- ⚽ **Match-Fixing Detection**: XGBoost model using match stats and betting patterns.
- 🌐 **Cyber Abuse Detection**: Fine-tuned RoBERTa model for text classification.
- 🔍 **Explainable AI (XAI)**: SHAP and attention-based explanations for every prediction.
- 📊 **Performance Monitoring**: Accuracy, Precision, F1-score, ROC-AUC, and response time.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/fairplay-ai.git
cd fairplay-ai
pip install -r requirements.txt'

Ensure your .streamlit/secrets.toml contains:

toml
Copy
Edit
GROQ_API_KEY = "your_groq_api_key"

📁 Project Structure
bash
Copy
Edit
fairplay-ai/
├── app.py                     # Streamlit UI
├── task_agent.py              # Query handling and intent routing
├── bio_agent.py               # Doping detection
├── match_fixing_agent.py      # Match-fixing detection
├── cyber_agent.py             # Cyber/social violation detection
├── models/                    # Trained models & scalers
├── data/                      # Sample datasets
└── .streamlit/secrets.toml    # API keys (not version-controlled)

