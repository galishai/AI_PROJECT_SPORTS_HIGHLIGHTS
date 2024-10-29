import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

file_path = 'output_v3_with_target_vf.csv'
dataset = pd.read_csv(file_path)

data = dataset.drop(columns=['target'])
target = dataset['target']
#print("Data type of 'games_played':", data['time_left_qtr'].dtype)


#encoding categorical columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

#splitting
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# probabilities of samples being 1
y_proba = rf.predict_proba(X_test)
# Get the probability of each sample being class 1
y_proba_1 = y_proba[:, 1]  # This selects the second column, which corresponds to class 1




