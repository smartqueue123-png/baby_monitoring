import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load the data you just recorded
df = pd.read_csv('training_data.csv')

# Initialize the Model
# Contamination is low because our training data is 100% "Normal"
model = IsolationForest(contamination=0.01)

# Train the model
model.fit(df)

# Save the "Brain" to a file
joblib.dump(model, 'baby_model.pkl')
print("Model trained and saved as baby_model.pkl!")