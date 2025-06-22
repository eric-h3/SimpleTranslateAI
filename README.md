# SimpleTranslateAI

SimpleTranslateAI is a machine translation app that translates text English to Chinese using transformer models. 

## Features

- Translate text from English to Chinese
- Fine-tune pre-trained translation models on custom datasets
- Evaluate translation quality with BLEU score and other metrics
- Easily extendable for other language pairs

## Models and Datasets

- **Pretrained Model:**  
  [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)  
  Used as the base model for Chinese-to-English translation.

- **Training Dataset:**  
  [FradSer/OpenSubtitles-en-zh-cn-20m](https://huggingface.co/datasets/FradSer/OpenSubtitles-en-zh-cn-20m)  
  Large-scale parallel corpus of English and Chinese subtitles, used for fine-tuning and evaluation.

## Project Structure

- `app/translate.py` — Main translation script
- `notebooks/train.ipynb` — Jupyter notebook for data preprocessing, training, and evaluation
- `model/` — Saved model files (not included in version control)
- `notebooks/finetuned-nlp-en-zh/` — Training checkpoints

## How to Get Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training notebook in `notebooks/train.ipynb` to fine-tune or evaluate a model.

3. Run `app/translate.py` to launch the gradio app to translate text using your fine-tuned model.

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets