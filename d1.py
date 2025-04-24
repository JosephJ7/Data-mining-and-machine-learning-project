import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from nltk.corpus import stopwords
import string
import nltk
from collections import Counter

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# 1. Introduction and Dataset Overview
# ======================================
# Dataset Name: Real or Fake Job Posting Prediction
# Source: Kaggle (https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data)
# Purpose: To predict whether a job posting is real or fake.
#
# Problem Type: Binary Classification
# Target Variable: fraudulent (1: Fake, 0: Real)
# Key Features: title, company_profile, description, requirements, benefits, employment_type, etc.

# Load the dataset
try:
    df = pd.read_csv("fake_job_postings.csv")  # Load from local file
except FileNotFoundError:
    # If the file is not found locally, attempt to load it from the Kaggle URL.  This will likely fail.
    print("Error: 'fake_job_postings.csv' not found. Please download it from Kaggle and place it in the same directory as this script.")
    df = None # set df to None to prevent further errors
    exit()

# Check for balanced or imbalanced class distribution
print("\nClass Distribution:")
print(df['fraudulent'].value_counts())
print(f"\nProportion of Fake Postings: {df['fraudulent'].mean():.4f}")

# 2. Data Preprocessing and Cleaning
# ======================================

# Initial Data Exploration
print("\nInitial Data Exploration:")
print(df.head())
print(df.info())

# Dropping unwanted columns
unwanted_columns = ['job_id']  # Removed location - high cardinality, difficult to process
df = df.drop(columns=unwanted_columns, errors='ignore')
print("\nAfter Dropping Unwanted Columns:")
print(df.head())

# Handling Missing Data
print("\nMissing Data Before Handling:")
print(df.isnull().sum())

# Impute missing text data with "Missing"
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
df[text_columns] = df[text_columns].fillna('Missing')

# Impute categorical features with the mode
categorical_columns = ['employment_type', 'required_education', 'required_experience', 'functional_area', 'industry']
for col in categorical_columns:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

# Impute numerical features (if any) with the median (if any numerical columns exist)
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('fraudulent') # remove the target variable
if numerical_cols: # Check if there are any numerical columns
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

print("\nMissing Data After Handling:")
print(df.isnull().sum())

# Feature Engineering & Transformation
def clean_text(text):
    """
    Cleans the input text by removing punctuation, converting to lowercase,
    and removing stop words.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        tokens = [w for w in tokens if not w in stop_words]
        return " ".join(tokens)
    else:
        return "" # Return empty string for non-string input


# Apply the cleaning function to the text columns
for col in text_columns:
    df[col] = df[col].apply(clean_text)

# Combine all text columns into a single text feature
df['combined_text'] = df[text_columns].apply(lambda row: ' '.join(row.values), axis=1)

# Vectorize the combined text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # Limiting features can improve performance
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate TF-IDF features with the original dataframe
df = pd.concat([df, tfidf_df], axis=1)

# Drop the original text columns and the combined text column
df = df.drop(columns=text_columns + ['combined_text'])

print("\nData After Text Vectorization:")
print(df.head())


# Handling Outliers (Z-score method)
print("\nOutlier Detection and Removal (Z-score):")
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('fraudulent') # Remove target
if numerical_cols:
    df_no_target = df[numerical_cols]
    z_scores = zscore(df_no_target)
    abs_z_scores = np.abs(z_scores)
    filtering_mask = (abs_z_scores < 3).all(axis=1) # boolean mask of rows with z-score < 3 for all cols
    df = df[filtering_mask] # Apply the mask
    print(f"Shape after outlier removal: {df.shape}")
else:
    print("No numerical columns for outlier detection.")

# 3. Exploratory Data Analysis (EDA)
# ======================================

print("\nExploratory Data Analysis:")

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Correlation Analysis (only for numeric columns)
numeric_df = df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')  # Removed annot=True for better visualization
plt.title('Correlation Matrix')
plt.show()

# Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='fraudulent', data=df)
plt.title('Distribution of Target Variable (Fraudulent)')
plt.xlabel('Fraudulent (0: Real, 1: Fake)')
plt.ylabel('Count')
plt.show()

# 4. Model Selection and Training
# ======================================
print("\nModel Selection and Training:")

# Separate features (X) and target variable (y)
X = df.drop(columns=['fraudulent'])
y = df['fraudulent']

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42) # 50% train
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, stratify=y_temp, random_state=42) # 30% val, 20% test

print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}, Testing set size: {len(X_test)}")


# Define models
logistic_regression_model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced') # Add class_weight
random_forest_model = RandomForestClassifier(random_state=42, class_weight='balanced') # Add class_weight

# K-fold cross-validation (Stratified)
def perform_cross_validation(model, X, y, cv=5):
    """
    Performs stratified k-fold cross-validation and returns the mean and standard deviation
    of the F1-score.

    Args:
        model: The machine learning model to evaluate.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        cv (int): The number of folds for cross-validation.  Defaults to 5.

    Returns:
        tuple: (mean_f1, std_f1) - The mean and standard deviation of the F1-score.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) # Use StratifiedKFold
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    return np.mean(f1_scores), np.std(f1_scores)

# Perform cross-validation for both models
print("\nCross-Validation Results:")
lr_mean_f1, lr_std_f1 = perform_cross_validation(logistic_regression_model, X_train, y_train)
print(f"Logistic Regression: Mean F1-score = {lr_mean_f1:.4f}, Std F1-score = {lr_std_f1:.4f}")

rf_mean_f1, rf_std_f1 = perform_cross_validation(random_forest_model, X_train, y_train)
print(f"Random Forest: Mean F1-score = {rf_mean_f1:.4f}, Std F1-score = {rf_std_f1:.4f}")



# Train the models
logistic_regression_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# 5. Model Performance Evaluation
# ======================================
print("\nModel Performance Evaluation:")

def evaluate_model(model, X, y, model_name="Model"):
    """
    Evaluates the performance of a given model and prints various metrics.
    Also generates and displays the confusion matrix and ROC curve.

    Args:
        model: The trained machine learning model.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        model_name (str): Name of the model
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix(y, y_pred), display_labels=[0, 1])
    confusion_matrix_display.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
    average_precision = average_precision_score(y, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall_curve, precision_curve, color='b',
             label=f'{model_name} (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()
    return y_pred, y_pred_proba #returning the predictions.


# Evaluate Logistic Regression
y_pred_lr, y_pred_proba_lr = evaluate_model(logistic_regression_model, X_test, y_test, "Logistic Regression")

# Evaluate Random Forest
y_pred_rf, y_pred_proba_rf = evaluate_model(random_forest_model, X_test, y_test, "Random Forest")



# 6. Compare Model Results
# ======================================
print("\nComparing Model Results:")

# Compare overall performance (Accuracy, F1, AUC)
results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'],
    'Logistic Regression': [
        accuracy_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_lr),
        roc_auc_score(y_test, y_pred_proba_lr)
    ],
    'Random Forest': [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf),
        roc_auc_score(y_test, y_pred_proba_rf)
    ]
}

results_df = pd.DataFrame(results)
print(results_df)
