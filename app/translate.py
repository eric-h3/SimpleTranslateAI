from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gradio as gr

# Load from saved directory
tokenizer = AutoTokenizer.from_pretrained("../model")
model = AutoModelForSeq2SeqLM.from_pretrained("../model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated_tokens = model.generate(**inputs, max_length=256)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

interface = gr.Interface(fn=translate,
                         inputs=gr.Textbox(lines=2, placeholder='Text to translate'),
                         outputs='text')

interface.launch()
