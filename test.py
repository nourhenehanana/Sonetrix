import whisper

model= whisper.load_model("small")
result= model.transcribe("Recording.mp3")
print(result["text"])