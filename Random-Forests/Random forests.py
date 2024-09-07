from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('fraud.csv')

# Preprocess data
X = df[['Transaction Amount', 'Transaction Time']]  # Example features
y = df['Fraud']  # Target (binary: 0 or 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
