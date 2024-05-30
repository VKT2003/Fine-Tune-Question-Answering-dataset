# Fine-Tuning Question Answering Model on Custom Dataset

This repository provides the code and instructions for fine-tuning a question answering (QA) model such as BERT, RoBERTa, or similar models on a custom dataset. The fine-tuning process aims to adapt the pre-trained QA model to effectively answer questions based on the specific context provided in your dataset.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Dataset](#dataset)
- [Fine-Tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Question answering models are designed to answer questions based on a given context. Fine-tuning pre-trained models like BERT, RoBERTa, or DistilBERT on a specific QA dataset can significantly improve their performance for targeted applications.

## Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- PyTorch 1.8.1 or higher
- Transformers library by Hugging Face

## Setup

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/qa-finetune.git
    cd qa-finetune
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Prepare your custom QA dataset in JSON format, following the SQuAD (Stanford Question Answering Dataset) format. Place the dataset in the `data/` directory:

```
data/
  train.json
  val.json
```

Ensure your JSON files follow this structure:

```json
{
  "data": [
    {
      "title": "Dataset Title",
      "paragraphs": [
        {
          "context": "Context paragraph where the answer is located.",
          "qas": [
            {
              "question": "What is the question?",
              "id": "unique-id-1",
              "answers": [
                {
                  "text": "Answer to the question",
                  "answer_start": 34
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## Fine-Tuning

To fine-tune the QA model on your dataset, execute the following command:

```bash
python fine_tune.py --train_file data/train.json --val_file data/val.json --model_name bert-base-uncased --output_dir models/qa-finetuned
```

### Fine-Tuning Parameters

- `--train_file`: Path to the training dataset file.
- `--val_file`: Path to the validation dataset file.
- `--model_name`: Name or path of the pre-trained model to use (e.g., `bert-base-uncased`).
- `--output_dir`: Directory where the fine-tuned model will be saved.

Additional training parameters such as batch size, learning rate, and number of epochs can be customized in the `fine_tune.py` script.

## Evaluation

After fine-tuning, evaluate the model to ensure it meets the desired performance metrics. Run the evaluation script as follows:

```bash
python evaluate.py --model_dir models/qa-finetuned --test_file data/val.json
```

This script will generate performance metrics relevant to QA tasks, such as exact match (EM) and F1 score.

## Results

The results of the fine-tuning process, including training and evaluation metrics, will be saved in the `results/` directory. Key metrics to consider are exact match (EM) and F1 score.

## Usage

To use the fine-tuned QA model for inference, load it using the Transformers library:

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/qa-finetuned")
model = AutoModelForQuestionAnswering.from_pretrained("models/qa-finetuned")

context = "Context paragraph where the answer is located."
question = "What is the question?"

inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]

outputs = model(**inputs)
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

Thank you for your interest in this project! If you have any questions or feedback, please open an issue or contact the repository maintainer.
