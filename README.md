# Financial Named Entity Recognition (NER) Project

A comprehensive machine learning project for extracting financial entities from news articles and documents using state-of-the-art NLP techniques.

## Project Overview

This project implements a domain-specific Named Entity Recognition system designed to identify and classify key financial entities in news articles and financial documents. The model is trained on the specialized Fin-NER corpus to ensure high accuracy in financial contexts.

### Key Features

- **Domain-Specific**: Optimized for financial news and documents
- **Five Entity Types**: ORGANIZATION, MONEY, DATE, PRODUCT, TICKER
- **Complete Pipeline**: Data curation → Training → Evaluation → Deployment
- **Ready-to-Deploy**: Includes web application for real-time inference
- **Performance Metrics**: Comprehensive evaluation with precision, recall, and F1-scores

## Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd financial-ner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## 📁 Project Structure

```
.
├── data/
│   ├── Fin-NER/
│   │   ├── train.csv
│   │   └── test.csv
│   └── README.md (Brief description of the dataset)
├── notebooks/
│   ├── 1_Data_Exploration_and_Preprocessing.ipynb
│   └── 2_Model_Training_and_Evaluation.ipynb
├── models/
│   └── (trained model files will be saved here)
├── app/
│   ├── app.py (Simple Streamlit or Flask app)
│   └── requirements.txt
├── .gitignore
├── requirements.txt
└── README.md
```
