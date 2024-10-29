import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

file_path = 'output_v3_with_target_vf.csv'
dataset = pd.read_csv(file_path)

original_data = dataset.copy()

#data = dataset.drop(columns=['is_highlight'])
#target = dataset['is_highlight']
#print("Data type of 'games_played':", data['time_left_qtr'].dtype)

train_data = dataset.iloc[378:]  # All rows except the last
test_data = dataset.iloc[:378]   # Only the last row

original_test_data = test_data.copy()


#encoding categorical columns
label_encoders = {}
for column in dataset.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    dataset[column] = label_encoders[column].fit_transform(dataset[column])

train_data = dataset.iloc[378:]  # All rows except the last
test_data = dataset.iloc[:378]   # Only the last row

#splitting
#X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

X_train = train_data.drop(columns=['is_highlight'])
y_train = train_data['is_highlight']
X_test = test_data[X_train.columns]  # Ensures we have the same columns

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
'''
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

probability_df = original_data.iloc[X_test.index].copy()  # Get non-encoded samples
probability_df['probability_class_1'] = y_proba_1  # Add probabilities as a new column
top_100 = probability_df.sort_values(by='probability_class_1', ascending=False).head(100)

# Print the top 100 samples with the highest probability of being class 1
print("Top 100 samples with the highest probability of being class 1:")
print(top_100)
'''

y_test_proba = rf.predict_proba(X_test)[:, 1]  # Probability for class 1

# Add the prediction to the original test data for interpretation
original_test_data['probability_class_1'] = y_test_proba

sorted_probabilities = original_test_data.sort_values(by='probability_class_1', ascending=False)
# Display the probability for the last game
print("Probability of plays being class 1:")
with pd.option_context('display.max_rows', None):
    print("Sorted probabilities for the last game:")
    print(sorted_probabilities)



