import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from datasets import Dataset, load_dataset
import numpy as np
import json
import os
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
class FinancialNER:

    def __init__(self, model_name = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        self.max_length = 128
    

    def load_data(self, data_file = "finer139_processed.json"):

        with open(data_file, 'r') as f:
            data = json.load(f)
        
        all_labels = set()

        for split in ['train', 'validation', 'test']:
            for example in data[split]:
                all_labels.update(example['labels'])
        
        all_labels = sorted(list(all_labels))
        if '0' in all_labels:
            all_labels.remove('0')
            all_labels = ['0'] + all_labels
        
        self.label2id = {label: i for i, label in enumerate(all_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        print(f"   Train: {len(data['train'])} sentences")
        print(f"   Validation: {len(data['validation'])} sentences") 
        print(f"   Test: {len(data['test'])} sentences")
        print(f"   Labels: {len(self.label2id)} unique labels")

        return data
    

    def tokenize_and_align_labels(self, examples):
        
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation = True,
            is_split_into_words = True,
            max_length = self.max_length,
            padding = True
        )

        labels = []

        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index = i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:                # Special tokens
                    label_ids.append(-100)
                elif word_idx is not previous_word_idx:
                    if word_idx < len(label):       # Firs
                        label_ids.append(self.label2id[label[word_idx]])
                    else:
                        label_ids.append(self.label2id['0'])
                else:
                    label_ids.append(-100)          # Continuation of the word (subword)
                
                previous_word_idx = word_idx
            
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs  


    def prepare_datasets(self, data):

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels = len(self.label2id),
            id2label = self.id2label,
            label2id = self.label2id
        )

        def create_dataset(split_data):
            return Dataset.from_dict({
                "tokens": [example["tokens"] for example in split_data],
                "labels": [example["labels"] for example in split_data]
            })

        train_dataset = create_dataset(data["train"])
        test_dataset = create_dataset(data["test"])
        val_dataset = create_dataset(data["validation"])

        train_dataset = train_dataset.map(self.tokenize_and_align_labels, batched = True)
        val_dataset = val_dataset.map(self.tokenize_and_align_labels, batched = True)
        test_dataset = test_dataset.map(self.tokenize_and_align_labels, batched = True)

        return train_dataset, val_dataset, test_dataset

    
    def compute_metrics(self, eval_pred):                   # Compute evaluation metrics

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.id2label[p] for (p, l) in zip(predictions, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [self.id2label[l] for (p, l) in zip(predictions, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        flat_true_labels = [label for sublist in true_labels for label in sublist]
        flat_predictions = [pred for sublist in true_predictions for pred in sublist]

        accuracy = accuracy_score(flat_true_labels, flat_predictions)

        return {
            "accuracy": accuracy
        }


    def train_model(self, train_dataset, val_dataset, output_dir = "./financial_ner_model"):

        training_args = TrainingArguments(
            output_dir = output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size= 16,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy= "epoch",
            save_strategy= "epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_steps=100,
            save_total_limit=2
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding = True
        )

        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer = self.tokenizer,
            data_collator = data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        trainer.save_model()

        self.tokenizer.save_pretrained(output_dir)

        return trainer
    

    def evaluate_model(self, trainer, test_dataset):

        results = trainer.evaluate(test_dataset)

        print("Test Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return results
    

    def predict_sample(self, text, model_dir="./financial_ner_model"):

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)

        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
        
        inputs = tokenizer(tokens, is_split_into_words = True, return_tensors = "pt", truncation = True)

        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)

        # Align predictions with original tokens
        word_ids = inputs.word_ids()
        predicted_labels = []
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is not None and word_idx != previous_word_idx:
                predicted_labels.append(self.id2label[predictions[0][len(predicted_labels)].item()])
            previous_word_idx = word_idx
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 40)
        for token, label in zip(tokens, predicted_labels):
            print(f"{token:15} -> {label}")
        
        return list(zip(tokens, predicted_labels))
    


def train_financial_ner():

    ner_trainer = FinancialNER()

    data = ner_trainer.load_data()

    train_dataset, val_dataset, test_dataset = ner_trainer.prepare_datasets(data)

    trainer = ner_trainer.train_model(train_dataset, val_dataset)

    results = ner_trainer.evaluate_model(trainer, test_dataset)

    print("\nTraining completed successfully!")
    print("Model saved to: ./financial_ner_model")
    
    return ner_trainer, trainer, results

def quick_prediction_test():
    
    print("\nTesting trained model...")
    
    ner_trainer = FinancialNER()
    
    # Sample financial sentences
    test_sentences = [
        "Apple Inc reported revenue of 394 billion dollars in fiscal year 2023",
        "The company's market cap reached 3 trillion dollars last quarter",
        "EBITDA increased by 15 percent compared to the previous year"
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n--- Sample {i} ---")
        ner_trainer.predict_sample(sentence)


if __name__ == "__main__":

    ner_trainer, trainer, results = train_financial_ner()

    quick_prediction_test()















