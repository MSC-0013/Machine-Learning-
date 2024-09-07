import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Load dataset (for example purposes, using a random dataset)
df = pd.read_csv('your_dataset.csv')

# Preprocess data
X = df[['Feature1', 'Feature2']]  # Replace with your features
y = df['Target']  # Replace with your target variable

# Initialize the model
model = RandomForestClassifier()

# Define K-Fold Cross-Validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=kf)

# Output the cross-validation results
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
print("Standard Deviation of CV Score:", np.std(cv_scores))
