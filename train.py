# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# SAMPLE dataset (you can replace with your health dataset)
data = {
    'fever': [98, 101, 102, 99, 100, 103, 97, 105],
    'cough': [1, 1, 0, 0, 1, 1, 0, 1],
    'breath_shortness': [0, 1, 1, 0, 0, 1, 0, 1],
    'infection': [0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['fever', 'cough', 'breath_shortness']]
y = df['infection']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")