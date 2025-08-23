import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from datasets import Dataset, load_dataset
import numpy as np
import json
import os
from transformers import (
    Autotokenizer, 
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

        self.tokenizer = Autotokenizer.from_pretrained(self.model_name)
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
                    














model_processor = FinancialNER()

model_processor.load_data("finer139_processed.json")

