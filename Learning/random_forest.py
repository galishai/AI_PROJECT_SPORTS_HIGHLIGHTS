import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import csv
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#0 for classification, 1 for probabilities
PROBABILITIES = 1

#379 first game
file_path = '/full season data/new_output_labeled.csv'

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




#value_counts = dataset['current_team'].value_counts()
#print(value_counts)

#encoding quarter column
dataset = pd.get_dummies(dataset, columns=['quarter'])
'''
quarter_encoder = LabelEncoder()
quarters = dataset['quarter'].tolist()
dataset['quarter'] = quarter_encoder.fit_transform(quarters)
'''
dataset['time_left_qtr'] = dataset['time_left_qtr'].apply(time_to_seconds)
dataset = dataset.drop(columns=['date'])

dataset = pd.get_dummies(dataset, columns=['home_team','away_team','current_team'])
'''
# encoding team columns
team_name_encoder = OneHotEncoder()
teams = dataset['home_team'].tolist() + dataset['away_team'].tolist() + dataset['current_team'].tolist()
team_name_encoder.fit(teams)
dataset['home_team'] = team_name_encoder.transform(dataset['home_team'])
dataset['away_team'] = team_name_encoder.transform(dataset['away_team'])
dataset['current_team'] = team_name_encoder.transform(dataset['current_team'])
'''
dataset = pd.get_dummies(dataset, columns=['name','assister','stolen_by'])
'''
#encoding player columns
player_name_encoder = OneHotEncoder()
players = dataset['name'].tolist() + dataset['assister'].tolist() + dataset['stolen_by'].tolist()
player_name_encoder.fit(players)
dataset['name'] = player_name_encoder.transform(dataset['name'])
dataset['assister'] = player_name_encoder.transform(dataset['assister'])
dataset['stolen_by'] = player_name_encoder.transform(dataset['stolen_by'])

#encoding stage
stage_encoder = LabelEncoder()
stages = dataset['stage'].tolist()
modified_dataset['stage'] = stage_encoder.fit_transform(stages)
data = modified_dataset.drop(columns=['is_highlight'])
target = modified_dataset[['is_highlight']]
#print("Data type of 'games_played':", data['time_left_qtr'].dtype)
X_train = data.iloc[TEST_LAST_ROW_CSV - 1:]
Y_train = target.iloc[TEST_LAST_ROW_CSV - 1:]
X_test = data.iloc[:TEST_LAST_ROW_CSV - 1]
Y_test = target.iloc[:TEST_LAST_ROW_CSV - 1]
'''
train_set, test_set = train_test_split(dataset, test_size=0.2, shuffle=False)
X_train = train_set.drop(columns = ['is_highlight'])
y_train = train_set['is_highlight']

X_test = test_set.drop(columns = ['is_highlight'])
y_test = test_set['is_highlight']
'''
new_y_train = []
for val in Y_train['is_highlight']:
    if val == 0:
        new_y_train.append(np.random.randint(2, 20))
    else:
        new_y_train.append(1)
Y_train['is_highlight'] = new_y_train

test_non_encode = dataset.iloc[:TEST_LAST_ROW_CSV - 1]
'''
'''
importances = rf.feature_importances_
feature_importances = sorted(zip(data.columns, importances), key=lambda x: x[1], reverse=True)
for feature, importance in feature_importances:
    print(f"Feature: {feature}, Importance: {importance}")
'''
for depth in [1,3,5,10,15,20,40,60,80]:
    rf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1, max_depth=depth) #class_weight='balanced')
    rf.fit(X_train, y_train)
    if PROBABILITIES:
        y_probs_train = rf.predict_proba(X_train)[:, 1]  # Probabilities for class 1
        y_probs_test = rf.predict_proba(X_test)[:, 1]
        for i in range(1):
        # Define a custom threshold
            threshold = 0.15

        # Apply the threshold
            train_pred_threshold = (y_probs_train >= threshold).astype(int)
            test_pred_threshold = (y_probs_test >= threshold).astype(int)

            # Evaluate on training data
            print(f"On training with threshold={threshold}:")
            print("Accuracy:", accuracy_score(y_train, train_pred_threshold))
            print("Classification Report:\n", classification_report(y_train, train_pred_threshold))
            print("Confusion Matrix:\n", confusion_matrix(y_train, train_pred_threshold))


            # Evaluate on test data
            print(f"On test with threshold={threshold}:")
            print("Accuracy:", accuracy_score(y_test, test_pred_threshold))
            print("Classification Report:\n", classification_report(y_test, test_pred_threshold))
            print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred_threshold))


            #sorted_probabilities = test_non_encode.sort_values(by='probability_class_1', ascending=False)
            #sorted_probabilities.to_csv('output_test_sorted_probabilities.csv', index=False)
        print("OK :)")
    else:
        importances = rf.feature_importances_
        feature_importances = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importances:
            print(f"Feature: {feature}, Importance: {importance}")
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)
        # Evaluate
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        print("On training:")
        print("Accuracy:", train_accuracy)
        print("Classification Report:\n", classification_report(y_train, train_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_train, train_pred))
        print("On test:")
        print("Accuracy:", test_accuracy)
        print("Classification Report:\n", classification_report(y_test, test_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))
        print("OK :)")







