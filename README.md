# Financial NER with FiNER-139 Dataset

A Financial Named Entity Recognition (NER) project using the FiNER-139 dataset, which contains financial texts annotated with 139 different financial entity types.

## Dataset Overview

**FiNER-139** is a specialized financial NER dataset containing:
- Financial documents from US company reports
- 139 different financial entity types (revenue, assets, liabilities, etc.)
- XBRL-based annotations for numerical entities
- Focus on context-dependent financial terms

### Dataset Statistics

<!-- Add your dataset statistics here -->
```
Dataset Splits:
    train       : 900,384 sentences, 40,827,499 tokens
    validation  : 112,494 sentences, 5,194,320 tokens
    test        : 108,378 sentences, 5,120,260 tokens
    Total       : 1,121,256 sentences, 51,142,079 tokens

Entity Analysis:
    Non-entity tokens (O): 0
    Entity tokens: 387,009
    Unique entity types: 169

Top 10 entity types:
  B-DebtInstrumentInterestRateStatedPercentage: 18,448
  B-LineOfCreditFacilityMaximumBorrowingCapacity: 14,730
  B-DebtInstrumentBasisSpreadOnVariableRate1: 14,469
  B-DebtInstrumentFaceAmount: 13,158
  B-AllocatedShareBasedCompensationExpense: 10,160
  B-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount: 9,162
  B-EffectiveIncomeTaxRateContinuingOperations: 8,684
  B-AmortizationOfIntangibleAssets: 7,458
  B-ConcentrationRiskPercentage1: 6,779
  B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod: 5,763   
```

## 🔍 Data Exploration

### Dataset Distribution
<!-- Add your dataset distribution plot here -->
![alt text](plots/image.png)

### Entity Analysis
<!-- Add your entity analysis plots here -->
![alt text](plots/image-1.png)

### Sentence Length Distribution
<!-- Add your sentence length plot here -->
![alt text](plots/image-2.png)

### Top Entity Types
<!-- Add your top entity types plot here -->
![alt text](plots/image-3.png)


## Quick Start

### 1. Data Preprocessing

```python
from finer_preprocessing import SimpleFiNERProcessor

processor = FiNERProcessor()
processor.run_complete_analysis()

processor = quick_load_and_explore()
```

### 2. Load Processed Data

```python
import json

with open('finer139_processed.json', 'r') as f:
    data = json.load(f)

print(f"Training sentences: {len(data['train'])}")
print(f"Labels available: {list(data['label_mapping'].keys())}")
```

### 3. Train NER Model

```python
from ner_training import train_financial_ner

ner_trainer, trainer, results = train_financial_ner()

```

### 4. Make Predictions

```python
from ner_training import FinancialNER

ner = FinancialNER()
predictions = ner.predict_sample("Apple Inc reported revenue of 394 billion dollars")

# Output: [('Apple', 'B-ORG'), ('Inc', 'I-ORG'), ('reported', 'O'), 
#          ('revenue', 'B-REVENUE'), ('of', 'O'), ...]
```

## 📁 Project Structure

```
financial-ner/
├── README.md
├── data_exploration_and_preprocessing.py       # Data preprocessing script
├── model_training.py                             # NER model training script
├── finer139_processed.json                     # Processed dataset
├── plots/                                      # Data exploration plots
│   ├── dataset_distribution.png
│   ├── entity_distribution.png
│   ├── sentence_lengths.png
│   └── top_entities.png
└── financial_ner_model/                        # Trained model directory

```

## Requirements

```bash
pip install datasets pandas matplotlib json
```

## Entity Types Examples

The FiNER-139 dataset includes financial entities such as:

- **Financial Metrics**: Revenue, Assets, Liabilities, EBITDA
- **Numerical Values**: Share counts, Market capitalization
- **Financial Ratios**: P/E ratio, Debt-to-equity ratio
- **Time Periods**: Quarterly reports, Fiscal years
- **Financial Instruments**: Stocks, Bonds, Derivatives

### Sample Annotations

```
Text   : The company reported revenue of $2.5 billion for Q3 2023
Labels : O O O revenue O O O O O O

Text   : Apple Inc's market cap reached $3 trillion dollars
Labels : ORG O O O O O O O
```

## Key Features

- **Financial Focus**: Specialized for financial document processing
- **Rich Annotations**: 139 different financial entity types
- **Large Scale**: Over 1M annotated sentences
- **Real-world Data**: From actual US company reports
- **Context-aware**: Handles context-dependent financial terms

## Model Development

- **✅ Data Preprocessing**: Complete data exploration and preprocessing pipeline
- **✅ BERT-based NER Model**: Fine-tuned BERT model for financial entity recognition
- **✅ Training Pipeline**: Automated training with evaluation metrics
- **✅ Prediction Interface**: Easy-to-use prediction API for new text
- **✅ Model Persistence**: Save/load trained models

### Model Architecture

- **Base Model**: BERT-base-uncased
- **Task**: Token Classification (NER)
- **Labels**: 169 financial entity types
- **Training**: 3 epochs with learning rate 2e-5
- **Evaluation**: Accuracy and F1-score metrics

### Training Results

```
Model Performance:
- Training Dataset: 900,384 sentences
- Validation Dataset: 112,494 sentences  
- Test Dataset: 108,378 sentences
- Model Size: ~110M parameters
- Training Time: 0.2 hours on GPU
```

### Usage Examples

```python
python ner_training.py

from ner_training import SimpleFinancialNER
trainer = SimpleFinancialNER("bert-base-uncased")

trainer.predict_sample("Tesla's market cap exceeded 800 billion")
```

## Dataset Source

- **Hugging Face**: [nlpaueb/finer-139](https://huggingface.co/datasets/nlpaueb/finer-139)
- **Zenodo**: [https://zenodo.org/record/6339605](https://zenodo.org/record/6339605)
- **Paper**: [FiNER-139: A Nested Named Entity Recognition Dataset for Financial Documents](https://arxiv.org/abs/2302.11157)
