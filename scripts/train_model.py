from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split (optional, we use all data here)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model
with open('artifacts/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save target names too (so we know what 0, 1, 2 mean)
with open('artifacts/target_names.pkl', 'wb') as f:
    pickle.dump(iris.target_names, f)
