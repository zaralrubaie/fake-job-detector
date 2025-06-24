#  Fake Job Detector

A powerful web app that detects **fake job postings** using a blend of **LLMs**, **TinyBERT**, and a custom feature-based classifier.

---

##  Live Demo
👉 [Try it here]((https://huggingface.co/spaces/zahraa12355/Fakenews_detector))


##  About the Project
This project aims to fight online recruitment fraud by automatically identifying fake job listings using:

-  **TinyBERT**: A pretrained transformer model optimized for small devices and fast inference.
- **CustomTinyBERTClassifier**: A lightweight classifier that adds domain-specific features (e.g., salary inconsistencies, suspicious keywords, formatting patterns).
-  **LLM-enhanced pipeline**: Boosts performance with deeper language understanding and cross-feature correlation.

---

## Features

- Detects fake job posts using both raw text and additional metadata
- Fast inference using TinyBERT
- Combines LLM embeddings + custom feature engineering
- Simple, clean interface (built with Streamlit / Gradio)

---
[![Hugging Face Spaces](https://img.shields.io/badge/Live-Demo-blue?logo=huggingface)](https://huggingface.co/spaces/zahraa12355/Fakenews_detector)

---
## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/zaralrubaie/fake-job-detector.git
cd fake-job-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```
