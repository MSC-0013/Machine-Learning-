from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('customers.csv')

# Preprocess data
X = df[['Annual Income', 'Spending Score']]  # Example features

# Initialize model
kmeans = KMeans(n_clusters=3, random_state=42)

# Train model
kmeans.fit(X)

# Predict clusters
y_kmeans = kmeans.predict(X)

# Visualize clusters
plt.scatter(X['Annual Income'], X['Spending Score'], c=y_kmeans, cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Clusters')
plt.show()
