# Multi-class Classifier for Strength Training Routines

## Project Overview
This repository contains the code and Jupyter Notebooks for building a multi-class Deep Neural Network (DNN) classifier using Keras and TensorFlow. The model predicts the experience level of strength training routines (Novice, Intermediate, Advanced) based on physiological parameters (volume, load, intensity).

## Academic Context
This project was developed for the course **Text Mining and Machine Learning (MTAA)** at the Universidad Nacional del Sur (UNS). It emphasizes clean architecture, tensor dimensionality awareness during Pandas preprocessing, and model explainability using SHAP.

## Architecture & Workflow

### 1. Local Preprocessing
Data starts as an "exercise-per-row" dataset (600,000 records). 
Run the preprocessing notebook locally to aggregate the data into a "routine-per-row" format.
* Notebook: `notebooks/01_data_preprocessing_and_eda.ipynb`
* Modules used: `src/preprocessing.py`

### 2. Google Colab Execution
Upload the preprocessed `aggregated_routines.csv` and execute the following phases in Google Colab:
* **Model Training:** `notebooks/02_model_architecture_and_training.ipynb` - Builds the Keras network with academic justifications for layer and hyperparameter choices.
* **Explainability:** `notebooks/03_explainability_and_shap.ipynb` - Uses SHAP to generate summary plots and force plots, auditing the systemic load and volume importance.

## Installation for Local Setup
```bash
pip install -r requirements.txt
```
