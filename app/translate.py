from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("../model")
model = AutoModelForSeq2SeqLM.from_pretrained("../model")

def translate(text):
    input_text = f">>cmn_Hans<< {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(**inputs, max_length=256)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

interface = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=4, label="English Input"),
    outputs=gr.Textbox(lines=4, label="Chinese Translation"),
    title="English to Chinese Translator",
    description="Translate English text to Chinese using a fine-tuned Helsinki-NLP/opus-mt-en-zh model."
)

interface.launch()
