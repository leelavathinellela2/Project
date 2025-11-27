# train.py
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Example dummy dataset
# Columns: [fever, cough (0/1), breath (0/1)]
X = np.array([
    [98.0, 0, 0],
    [99.0, 0, 0],
    [100.5, 1, 0],
    [101.0, 1, 1],
    [102.2, 1, 1],
    [97.5, 0, 0],
    [103.0, 1, 1],
    [99.5, 0, 1],
    [100.0, 1, 0],
    [101.5, 1, 1]
])

# Labels: 1 = infected/high chance, 0 = low chance
y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])

# Train a simple classifier
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved model.pkl (size: {:.1f} KB)".format((open("model.pkl","rb").seek(0,2) or 0)/1024 if False else (0)))
# (The above print avoids reading file size reliably on all systems; you will see the file in your folder)
