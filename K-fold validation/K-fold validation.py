from sklearn.model_selection import cross_val_score

# Initialize model (using previous RandomForest model)
model = RandomForestClassifier()

# Perform k-fold cross validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())
