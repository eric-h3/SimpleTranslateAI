# SimpleTranslateAI

An implementation of machine translation using hugging Face's translation models. Runs and builds a fine-tune version of the Helsinki-NLP/opus-mt-en-zh model using the opus100 (en-zh). The model is then interactable using a simple translation app.

![Alt text](/images/translate.PNG)

## Features

- Translate text from English to Chinese
- Fine-tune pre-trained Helsinki translation models on custom datasets
- Evaluate translation quality with BLEU score and other metrics
- Easily extendable for other language pairs

## Models and Datasets

- **Base Model:**  
  [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)  
  Used as the base model for Chinese-to-English translation.

- **Dataset used to train**  
  [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)  
  OPUS-100 is an English-centric multilingual corpus covering 100 languages.

## Project Structure

- `notebooks/train.ipynb` — Jupyter notebook for data preprocessing, training, and evaluation
- `app/translate.py` — Main translation app
- `model/` — Saved model files (not included in version control)

## Training hyperparameters
- learning_rate: 1e-5
- train_batch_size: 32
- eval_batch_size: 32
- max steps: 3000
- optimizer: adamw_torch

## Training results
- training loss: 1.806
- validation loss: 1.679
- epoch: 0.096
- bleu: 15.159

## How to Get Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training notebook in `notebooks/train.ipynb` to fine-tune or evaluate a model.

3. Run `app/translate.py` to launch the gradio app to translate text using your fine-tuned model.

## Framework versions

- Python 3.10.11
- PyTorch 2.7.1+cu118
- Transformers 4.52.4
- Tokenizers 0.21.1
- Datasets 2.16.1
