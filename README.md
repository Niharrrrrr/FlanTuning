# FlanTuning

# FLAN-T5 Fine-Tuning on Multiple NLP Tasks

## Project Overview

This project focuses on fine-tuning the FLAN-T5 model (Large) across several Natural Language Processing (NLP) tasks, including text classification, question answering, question generation, paraphrasing, summarization, and semantic similarity. The model was fine-tuned on specific datasets for each task to enhance its performance in a variety of language understanding and generation tasks.

### Datasets Used for Fine-Tuning

1. **Text Classification** - [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
   - **Task**: Classifying the emotional content of text into categories like happiness, anger, sadness, etc.
   
2. **Question Answering** - [mlqa.hi.en](https://huggingface.co/datasets/mlqa)
   - **Task**: Multilingual question answering with a focus on Hindi-English translation.

3. **Question Generation** - [SQuAD](https://huggingface.co/datasets/squad)
   - **Task**: Generating questions based on given text passages.

4. **Paraphrasing** - [PAWS](https://huggingface.co/datasets/paws)
   - **Task**: Detecting and generating paraphrases for sentences.

5. **Summarization** - [SAMSum](https://huggingface.co/datasets/samsum)
   - **Task**: Summarizing conversations from chat-like dialogues.

6. **Semantic Similarity** - [PAWS](https://huggingface.co/datasets/paws)
   - **Task**: Determining whether two sentences are semantically similar.

Additionally, sentiment analysis was used as a comparison task between the base and fine-tuned models, showcasing how model performance varies across different tasks.

## Fine-Tuning Methodology

### Model: FLAN-T5 (Large)

The fine-tuning process involved supervised learning for each specific dataset to adapt the model’s general text generation and understanding capabilities to each task.

### Steps for Fine-Tuning:

1. **Data Preparation**:
   - Each dataset was preprocessed and formatted to fit the text-to-text paradigm used by the T5 model.
   - For instance, in the question answering task, inputs were formatted as `<context> <question>`, and outputs were the answer.

2. **Fine-Tuning Process**:
   - For each dataset, the model was initialized with pre-trained weights from `google/flan-t5-large`.
   - The fine-tuning was done using the Hugging Face `Trainer` API, performing supervised training on each task's dataset.
   - The model was optimized using AdamW, with regular evaluations on validation sets to monitor performance.

3. **Task-Specific Hyperparameters**:
   - **Batch Size**: Varied per dataset (8 or 16)
   - **Learning Rate**: 3e-5 for most tasks
   - **Epochs**: 3 to 5 depending on task complexity and dataset size
   - **Optimizer**: AdamW

4. **Hardware Acceleration**:
   - Fine-tuning was conducted on a GPU-enabled environment, leveraging Kaggle to efficiently train on large datasets.

### Sentiment Analysis Comparison

Although the core focus of this project was on the aforementioned tasks, a separate sentiment analysis comparison was conducted using the `dair-ai/emotion` dataset. This comparison illustrated how fine-tuning can improve task-specific performance.

## Inference Example (Text Classification Task)

Below is an example of how to use the fine-tuned model for text classification:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("your-fine-tuned-model")
model = T5ForConditionalGeneration.from_pretrained("your-fine-tuned-model", device_map="auto")

# Input sentence for classification
input_text = "What a beautiful day, I feel so happy!"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# Generate prediction
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```
