import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import csv

#0 for classification, 1 for probabilities
PROBABILITIES = 0


TEST_LAST_ROW_CSV = 3399
file_path = 'output_v3_with_target_vf.csv'


def seconds_to_time(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


dataset = pd.read_csv(file_path)
modified_dataset = dataset.copy()

# encoding categorical columns
label_encoders = {}
for column in modified_dataset.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    modified_dataset[column] = label_encoders[column].fit_transform(modified_dataset[column])

data = modified_dataset.drop(columns=['is_highlight', 'date'])
target = modified_dataset['is_highlight']
#print("Data type of 'games_played':", data['time_left_qtr'].dtype)
X_train = data.iloc[TEST_LAST_ROW_CSV - 1:]
Y_train = target.iloc[TEST_LAST_ROW_CSV - 1:]
X_test = data.iloc[:TEST_LAST_ROW_CSV - 1]
Y_test = target.iloc[:TEST_LAST_ROW_CSV - 1]

test_non_encode = dataset.iloc[:TEST_LAST_ROW_CSV - 1]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

if PROBABILITIES == 1:
    y_test_proba = rf.predict_proba(X_test)[:, 1]  # Probability for class 1
    test_non_encode['probability_class_1'] = y_test_proba

    sorted_probabilities = test_non_encode.sort_values(by='probability_class_1', ascending=False)
    sorted_probabilities['time_left_qtr'] = sorted_probabilities['time_left_qtr'].apply(seconds_to_time)
    sorted_probabilities.to_csv('output_test_sorted_probabilities.csv', index=False)
    print(f"Play component text saved to output_test_sorted_probabilities.csv")
    print("OK :)")
else:
    Y_pred = rf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

    test_non_encode['classified'] = Y_pred
    test_non_encode['time_left_qtr'] = test_non_encode['time_left_qtr'].apply(seconds_to_time)
    test_non_encode.to_csv('output_test_classified.csv', index=False)
    print("OK :)")








