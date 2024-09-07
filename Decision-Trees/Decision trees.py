from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('loans.csv')

# Preprocess data
X = df[['Income', 'Loan Amount']]  # Example features
y = df['Approved']  # Target (binary: 0 or 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print(classification_report(y_test, y_pred))
