from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv('movies.csv')

# Preprocess data
X = df[['Feature1', 'Feature2']]  # Example features
y = df['Genre']  # Target (categorical)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = KNeighborsClassifier(n_neighbors=5)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)