import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import csv

#0 for classification, 1 for probabilities
PROBABILITIES = 1

#379 first game
TEST_LAST_ROW_CSV = 379
file_path = 'output_v3_with_target_vf.csv'

def time_to_seconds(time_str):
    if ':' in time_str:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    else:
        return int(float(time_str))

def seconds_to_time(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


dataset = pd.read_csv(file_path)
modified_dataset = dataset.copy()

#value_counts = dataset['current_team'].value_counts()
#print(value_counts)

# encoding team columns
team_name_encoder = LabelEncoder()
teams = dataset['home_team'].tolist() + dataset['away_team'].tolist() + dataset['current_team'].tolist()
team_name_encoder.fit(teams)
modified_dataset['home_team'] = team_name_encoder.transform(modified_dataset['home_team'])
modified_dataset['away_team'] = team_name_encoder.transform(modified_dataset['away_team'])
modified_dataset['current_team'] = team_name_encoder.transform(modified_dataset['current_team'])

#encoding quarter column
quarter_encoder = LabelEncoder()
quarters = dataset['quarter'].tolist()
modified_dataset['quarter'] = quarter_encoder.fit_transform(quarters)

#encoding player columns
player_name_encoder = LabelEncoder()
players = dataset['name'].tolist() + dataset['assister'].tolist() + dataset['stolen_by'].tolist()
player_name_encoder.fit(players)
modified_dataset['name'] = player_name_encoder.transform(modified_dataset['name'])
modified_dataset['assister'] = player_name_encoder.transform(modified_dataset['assister'])
modified_dataset['stolen_by'] = player_name_encoder.transform(modified_dataset['stolen_by'])

#encoding stage
stage_encoder = LabelEncoder()
stages = dataset['stage'].tolist()
modified_dataset['stage'] = stage_encoder.fit_transform(stages)

modified_dataset['time_left_qtr'] = modified_dataset['time_left_qtr'].apply(time_to_seconds)
modified_dataset = modified_dataset.drop(columns=['date'])

data = modified_dataset.drop(columns=['is_highlight'])
target = modified_dataset[['is_highlight']]
#print("Data type of 'games_played':", data['time_left_qtr'].dtype)
X_train = data.iloc[TEST_LAST_ROW_CSV - 1:]
Y_train = target.iloc[TEST_LAST_ROW_CSV - 1:]
X_test = data.iloc[:TEST_LAST_ROW_CSV - 1]
Y_test = target.iloc[:TEST_LAST_ROW_CSV - 1]

new_y_train = []
for val in Y_train['is_highlight']:
    if val == 0:
        new_y_train.append(np.random.randint(2, 20))
    else:
        new_y_train.append(1)
Y_train['is_highlight'] = new_y_train

test_non_encode = dataset.iloc[:TEST_LAST_ROW_CSV - 1]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

importances = rf.feature_importances_
feature_importances = sorted(zip(data.columns, importances), key=lambda x: x[1], reverse=True)
for feature, importance in feature_importances:
    print(f"Feature: {feature}, Importance: {importance}")



if PROBABILITIES == 1:
    y_test_proba = rf.predict_proba(X_test)
    y_test_proba = rf.predict_proba(X_test)[:, 0]  # Probability for class 1
    test_non_encode['probability_class_1'] = y_test_proba

    sorted_probabilities = test_non_encode.sort_values(by='probability_class_1', ascending=False)
    sorted_probabilities.to_csv('output_test_sorted_probabilities.csv', index=False)
    print("OK :)")
else:
    Y_pred = rf.predict(X_test)
    new_y_pred = []
    for val in Y_pred:
        if val == 1:
            new_y_pred.append(1)
        else:
            new_y_pred.append(0)
    Y_pred = new_y_pred
    # Evaluate
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

    test_non_encode['classified'] = Y_pred
    test_non_encode.to_csv('output_test_classified.csv', index=False)
    print("OK :)")








