import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Optional
from datasets import load_dataset
import json


class FiNERProcessor:

    def __init__(self):
        self.dataset = None
        self.id2label = {}

    def download_and_load_dataset(self):
        self.dataset = load_dataset("nlpaueb/finer-139")

        label_feature = self.dataset['train'].features['ner_tags'].feature
        self.id2label = {i: label for i, label in enumerate(label_feature.names)}
        
        print(f"Available splits: {list(self.dataset.keys())}")
        for split in self.dataset.keys():
            print(f"   {split}: {len(self.dataset[split]):,} examples")


    def show_sample(self, num_samples = 3):

        for i in range(num_samples):
            example = self.dataset['train'][i]
            tokens = example['tokens']
            labels = [self.id2label[tag_id] for tag_id in example['ner_tags']]

            print(f"\nSample {i+1}: ")
            print("Text   :", " ".join(tokens))
            print("Labels :", " ".join(labels))


    def get_basic_stats(self):

        all_labels = []
        total_tokens = 0
        total_sentences = 0

        for split_name, split_data in self.dataset.items():
            sentences = len(split_data)
            tokens = sum(len(example['tokens']) for example in split_data)

            print(f"{split_name:12}: {sentences:,} sentences, {tokens:,} tokens")

            total_sentences += sentences
            total_tokens += tokens

            for example in split_data:
                labels = [self.id2label[tag_id] for tag_id in example['ner_tags']]
                all_labels.extend(labels)
        
        print(f"{'Total':12}: {total_sentences:,} sentences, {total_tokens:,} tokens")
        return all_labels


    def analyze_entities(self, all_labels):

        label_counts = Counter(all_labels)

        o_count = label_counts.get('0', 0)
        entity_labels = {k: v for k, v in label_counts.items() if k != 'O'}

        print(f"Non-entity tokens (O): {o_count:,}")
        print(f"Entity tokens: {sum(entity_labels.values()):,}")
        print(f"Unique entity types: {len(entity_labels)}")

        print(f"\nTop 10 entity types:")
        for label, count in Counter(entity_labels).most_common(10):
            print(f"  {label:15}: {count:,}")


    def create_plots(self, all_labels):

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('FiNER-139 Dataset Overview', fontsize=16)
        # Plot-1: Dataset overview
        split_names = list(self.dataset.keys())
        split_sizes = [len(self.dataset[split]) for split in split_names]
        
        axes[0,0].bar(split_names, split_sizes, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0,0].set_title('Dataset Splits')
        axes[0,0].set_ylabel('Number of Sentences')

        # Plot-2: Entity vs Non-Entity
        label_counts = Counter(all_labels)
        o_count = label_counts.get('O', 0)
        entity_count = sum(v for k, v in label_counts.items() if k != 'O')
        
        axes[0,1].pie([o_count, entity_count], 
                      labels=['Non-Entity', 'Entity'], 
                      autopct='%1.1f%%', 
                      colors=['lightgray', 'lightblue'])
        axes[0,1].set_title('Entity vs Non-Entity Tokens')

        # Plot-3: Sentence lengths
        train_lengths = [len(example['tokens']) for example in self.dataset['train']]
        
        axes[1,0].hist(train_lengths, bins=30, alpha=0.7, color='green')
        axes[1,0].set_title('Sentence Length Distribution')
        axes[1,0].set_xlabel('Number of Tokens')
        axes[1,0].set_ylabel('Frequency')

        # 4. Top entity types
        entity_labels = {k: v for k, v in label_counts.items() if k != 'O'}
        top_entities = Counter(entity_labels).most_common(10)
        
        if top_entities:
            labels, counts = zip(*top_entities)
            axes[1,1].barh(range(len(labels)), counts)
            axes[1,1].set_yticks(range(len(labels)))
            axes[1,1].set_yticklabels(labels)
            axes[1,1].set_title('Top 10 Entity Types')
            axes[1,1].set_xlabel('Count')
        
        plt.tight_layout()
        plt.show()


    def save_data(self, output_file = "finer139_processed.json"):

        processed_data = {
            'label_mapping': self.id2label,
            'train': [],
            'test': [],
            'validation': []
        }

        for split_name in self.dataset.keys():
            for example in self.dataset[split_name]:
                tokens = example['tokens']
                labels = [self.id2label[tag_id] for tag_id in example['ner_tags']]
                
                processed_data[split_name].append({
                    'tokens': tokens,
                    'labels': labels
                })
        
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)

    
    def run_complete_analysis(self):

        self.download_and_load_dataset()
        self.show_sample()

        all_labels = self.get_basic_stats()

        self.analyze_entities(all_labels)

        self.create_plots(all_labels)

        self.save_data()

    
def load_and_explore():
    processor = FiNERProcessor()
    processor.run_complete_analysis()
    return processor

def load_processed_data(file_path = "finer139_processed.json"):
    with open(file_path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    processor = load_and_explore()

