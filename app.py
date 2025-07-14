import whisper
from flask import Flask, request, render_template
import os
import re
from Translation_Task.MBart_text_translation import MBart_translation_task
from text_2_image.Stable_Diffusion import generate_image
import base64

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static"
model = whisper.load_model("small")

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files["file"]
        prompt = request.form.get("prompt", "")
        language = request.form.get("language", "")

        if file:
            filename = file.filename
            audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(audio_file_path)
            with open(audio_file_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')

            #if not re.match(r'^\w+$', prompt):
                #return render_template("index.html", error="The prompt must be a single word (no spaces).", prompt=prompt, language=language)'''


            #task 1: translation (get the language of the input and traslate using MBart)
            
            if prompt.lower() in ['translate', 'translation', 'tr', 'trl']:
                transcription = model.transcribe(audio_file_path)
                original_language = transcription["language"]
                translated_text = MBart_translation_task(original_language, language, transcription["text"])
                return render_template("index.html",audio_data=audio_data, result=translated_text, prompt=prompt, language=language)

            #task 2: transcription 
            elif prompt.lower() in ['transcript', 'transcription', 'trp', 'transcribe']:
                result = model.transcribe(audio_file_path)
                return render_template("index.html",audio_data=audio_data, result=result["text"], prompt=prompt, language=language)
            
            #task 3: image generation 
            elif prompt.lower() == 'generate an image':
                message = model.transcribe(audio_file_path)
                image_gen=generate_image(message["text"])
                return render_template("index.html",audio_data=audio_data, result=message["text"], image=image_gen, prompt=prompt, language=language)

            else:
                return render_template("index.html",audio_data=audio_data, error="The model cannot understand the task. Please enter 'translate' or 'transcribe'.", prompt=prompt, language=language)

    # GET method or no file uploaded
    return render_template("index.html", result=None, prompt="", language="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)  


