import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset
data = pd.DataFrame({
    'feature1': [2, 4, 6, 8],
    'feature2': [1, 3, 5, 7],
    'label': [0, 0, 1, 1]
})

X = data[['feature1', 'feature2']]
y = data['label']

model = LogisticRegression()
model.fit(X, y)

# Save the model
with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)
