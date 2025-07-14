# Sonetrix

🎙️📝🖼️ **Sonetrix** is an end‑to‑end Flask application that explores multimodal AI by transforming spoken audio into:
- Text transcription in the original language
- Multilingual translation (e.g., French, Italian)
- High‑resolution AI‑generated images that visually represent the audio content

## ✨ Features
- **Speech‑to‑Text**: Uses OpenAI Whisper for accurate audio transcription.
- **Multilingual Translation**: Integrates a fine‑tuned BART model to translate text into multiple target languages.
- **Image Generation**: Employs Dreamlike diffusion models to produce high‑quality images inspired by the audio's meaning.
- **Multimodal Pipeline**: Combines audio, text, and image generation into a seamless interactive web app.

## 🚀 Installation
Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies:

bash

pip install -r requirements.txt
# ⚙️ Usage
Start the Flask server:

bash
python app.py
Then, open your browser and navigate to:
http://127.0.0.1:5000
Upload an audio file and choose one of:

Transcribe: Convert speech to text.

Translate: Translate text into other languages.

Generate Image: Create an AI-generated visual based on the audio content.

🧩 Tech Stack
Flask — Web framework

OpenAI Whisper — Speech-to-text

BART (fine‑tuned) — Multilingual translation

Dreamlike Diffusion Models — Image generation


