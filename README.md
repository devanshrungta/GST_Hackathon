# Binary Classification Model for GST Analytics

## Overview
This project is a part of a hackathon aimed at developing AI/ML solutions for GST (Goods and Services Tax) analytics. The goal of the project is to predict a binary target using a large anonymized dataset that includes various features, some of which are categorical, numerical, and binary. The target variable represents a classification, and the challenge is to create a high-performing model using various machine learning techniques.

## Objective
The objective of this project is to build a machine learning model that provides accurate predictions for the target variable, with a particular focus on overcoming the class imbalance and achieving high accuracy. Our goal is to use a variety of machine learning algorithms, including ensemble methods, to tune hyperparameters and create an ensemble of the top models to optimize performance.

## Key Features
- Large dataset with 900,000+ rows and 21 columns.
- Various types of features: binary, categorical, numerical.
- Significant class imbalance in the target variable.
- Comprehensive data preprocessing and feature engineering to handle missing values, impute data, and scale features.
- Comparison of baseline models.
- Hyperparameter tuning using Optuna.
- Ensemble of top models for performance optimization.

## Workflow
The workflow for this project is structured as follows:

1. **Data Exploration and Preprocessing**:
   - Explored the dataset, identified missing values, and performed visualizations to understand data distributions.
   - Imputation of missing values using various strategies.
   - Feature engineering: transformed features for optimal model performance.
   
2. **Baseline Model Training**:
   - Trained 13 different baseline models including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, and others.
   - Evaluated the models based on metrics like Accuracy, Precision, Recall, F1-score, and AUC-ROC.

3. **Hyperparameter Tuning**:
   - Performed hyperparameter tuning using Optuna on the best performing models from the baseline.
   - Fine-tuned models include XGBoost, LightGBM, CatBoost, and others.
   - Evaluated hyperparameter-tuned models and visualized the results.

4. **Ensemble Techniques**:
   - Combined the top-performing models to improve overall performance using an ensemble of models.
   - Evaluated and compared the model combinations.

5. **Final Model Selection**:
   - Selected the best model based on performance metrics and saved the trained model as `LightGBM.pkl`.

---

## Installation
To set up the project environment, the following libraries must be installed using Python 3.x:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn optuna imbalanced-learn
```
Ensure that you have these additional libraries for plotting and visualization:
- matplotlib
- seaborn

## Project Structure
```
├── Baseline Models/                     
│   ├── baseline_model_plots
│   ├── model_comparison_plots
│   ├── Baseline_model_training.ipynb
│   ├── classification_metrics.json
├── Hyperparameter Tuning/                   
│   ├── Comparison Plots
│   ├── hypertuned_model_plots
│   ├── Notebooks
│   ├── optuna_visualizations
│   ├── Results
│   ├── CombiningModels.ipynb
│   ├── model_combinations_results.json
├── Trial Notebooks/                  
│   ├── data_preprocessing.py
│   ├── data_preprocessing.py
├── Citation and Plagiarism Declaration Report.pdf
├── LightGBM.pkl
├── LightGBM_confusion_matrix.png
├── LightGBM_roc_curve.png
├── lightgbm-optuna.ipynb
├── MODEL PERFORMANCE REPORT.pdf
├── submission.ipynb
```

The project is organized into the following directories and files:

1. **Baseline Models**: 
   - Trained 13 different models.
   - `baseline_model_plots/`: Contains the confusion matrices and ROC-AUC curves of all 13 models.
   - `model_comparison_plots/`: Contains comparison plots for different metrics across all models.
   - `Baseline_model_training.ipynb`: Jupyter notebook with the code for training the baseline models.
   - `classification_metrics.json`: JSON file containing the results and metrics of the trained models.

2. **Hyperparameter Tuning**:
   - Performed hyperparameter tuning on 7 different models.
   - `Comparison Plots/`: Comparison plots for different metrics across all hypertuned models.
   - `hypertuned_model_plots/`: Contains the confusion matrices and ROC-AUC curves of all 7 hypertuned models.
   - `Notebooks/`: Contains the code for hyperparameter tuning of each model in separate Jupyter notebooks.
   - `optuna_visualizations/`: Visualizations generated during Optuna-based hyperparameter tuning.
   - `Results/`: JSON file with results and metrics of hypertuned models.
   - `CombiningModels.ipynb`: Code implementing ensemble techniques to combine models and assess performance improvements.
   - `model_combinations_results.json`: JSON file containing the results of model combinations.

3. **Trial Notebooks**:
   - Contains Jupyter notebooks focused on feature engineering techniques.

4. **LightGBM.pkl**: 
   - The saved best hypertuned LightGBM model.

5. **LightGBM_confusion_matrix.png**: 
   - Confusion matrix of the best LightGBM model.

6. **LightGBM_roc_curve.png**: 
   - ROC curve of the best LightGBM model.

7. **lightgbm-optuna.ipynb**: 
   - Jupyter notebook with the code for hyperparameter tuning LightGBM using Optuna.

8. **submission.ipynb**: 
   - The final Jupyter notebook that includes EDA, imputation, preprocessing, and evaluation of model metrics.

Additionally, the project includes the following reports:
- `Citation and Plagiarism Declaration Report.pdf`
- `MODEL PERFORMANCE REPORT.pdf`

## Model Performance
After hyperparameter tuning, the following metrics were achieved by the best model:
- **Model**: LightGBM
- **Accuracy**: 97.86%
- **Precision**: 0.8508
- **Recall**: 0.9385
- **F1-score**: 0.8925
- **AUC-ROC**: 0.9948

The LightGBM model was selected as the final model based on its superior performance across all evaluation metrics.