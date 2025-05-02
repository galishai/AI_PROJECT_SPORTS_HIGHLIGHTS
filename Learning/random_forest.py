from tabnanny import verbose

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import csv
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import random
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

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

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_dataset(path, verbose=False):
    dataset = pd.read_csv(path)
    aug_count = 0
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1
    #quarter to categorical
    dataset = pd.get_dummies(dataset, columns=['quarter'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #time to ordinal
    dataset['time_left_qtr'] = dataset['time_left_qtr'].apply(time_to_seconds)
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #dates to ordinal
    dataset['date'] = pd.to_datetime(dataset['date'], format='%B %d, %Y')
    first_date = dataset['date'].min()
    dataset['days_since_first_game'] = (dataset['date'] - first_date).dt.days
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1
    dataset = dataset.drop(columns=['date'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #convert plays to categorical
    dataset = pd.get_dummies(dataset, columns=['play'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #teams to categorical
    dataset = pd.get_dummies(dataset, columns=['home_team', 'away_team', 'current_team'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    #player names to categorical
    dataset = pd.get_dummies(dataset, columns=['name', 'assister', 'stolen_by'])
    if verbose:
        print(f"{aug_count}. num cols: {len(dataset.columns)}")
        aug_count+=1

    return dataset

def main():
    #0 for classification, 1 for probabilities
    PROBABILITIES = 1

    #379 first game
    file_path = "../full season data/plays_with_onehot_v1.csv"

    frozen_seed = 42

    freeze_seeds(frozen_seed)

    nba_dataset = get_dataset(file_path)

    train_dataset, test_dataset = train_test_split(nba_dataset, test_size=0.2, shuffle=True, random_state=frozen_seed)

    #train_dataset, val_dataset = train_test_split(trainval_dataset, test_size=0.25, shuffle=True, random_state=frozen_seed)

    '''
    print(f'train: {len(train_dataset)}')
    print(f'val: {len(val_dataset)}')
    print(f'test: {len(test_dataset)}')
    '''

    class ThresholdClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, base_estimator=None, threshold=0.1):
            self.base_estimator = base_estimator
            self.threshold = threshold

        def fit(self, X, y):
            # Clone to ensure fresh estimator for each fit() call
            self.estimator_ = clone(self.base_estimator)
            self.estimator_.fit(X, y)
            return self

        def predict(self, X):
            proba = self.estimator_.predict_proba(X)[:, 1]
            return (proba >= self.threshold).astype(int)

        def predict_proba(self, X):
            return self.estimator_.predict_proba(X)


    X_train = train_dataset.drop(columns=['is_highlight'])
    y_train = train_dataset['is_highlight']
    #X_val = val_dataset.drop(columns=['is_highlight'])
    #y_val = val_dataset['is_highlight']
    X_test = test_dataset.drop(columns=['is_highlight'])
    y_test = test_dataset['is_highlight']

    thresholds = np.linspace(0.05, 0.5, 21)
    beta = 2.0 #Choose hyperparameter combo that maximizes F_beta

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=frozen_seed)
    scorer = make_scorer(fbeta_score, beta=beta)

    base_rf = RandomForestClassifier(random_state=42)
    thresh_clf = ThresholdClassifier(base_estimator=base_rf)

    depth_grid = {
        'base_estimator__max_depth': [3, 5, 7, 10]
    }

    grid_depth = GridSearchCV(
        estimator=thresh_clf,
        param_grid=depth_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    # Run grid search
    grid_depth.fit(X_train, y_train)

    best_depth = grid_depth.best_params_['base_estimator__max_depth']


    best_rf = RandomForestClassifier(max_depth=best_depth, random_state=frozen_seed)
    best_rf.fit(X_train, y_train)
    probs_val = best_rf.predict_proba(X_test)[:,1]
    best_thr, best_score = 0.1, -1
    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        score = fbeta_score(y_test, preds, beta=beta)
        if score > best_score:
            best_thr, best_score = t, score

    print(f"Depth={best_depth}, thr={best_thr:.2f} → Test F_{beta}={best_score:.3f}")

    '''
    best_params = {'score': -np.inf}
    results = []
    for depth in depths:
        print(f"Checking depth: {depth}")
        fold_scores = []
        fold_results = []
    
        for curr_fold_num, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            print(f"curr_fold_num: {curr_fold_num}")
            X_t = X_train.iloc[train_idx]
            X_v = X_train.iloc[val_idx]
            y_t = y_train.iloc[train_idx]
            y_v = y_train.iloc[val_idx]
    
            #print(X_t.head())
            #print(X_v.head())
            #print(y_t.head())
    
            # train
            classifier= RandomForestClassifier(max_depth=depth, random_state=42)
            classifier.fit(X_t, y_t)
    
            # get probabilities
            probs = classifier.predict_proba(X_v)[:, 1]
    
            # for each threshold, compute F_beta
            best_thresh_score = -np.inf
            best_thresh = None
            for thresh in thresholds:
                predictions = (probs >= thresh).astype(int)
                score = fbeta_score(y_v, predictions, beta=beta)
                if score > best_thresh_score:
                    best_thresh_score = score
                    best_thr = thresh
                curr_res = {'max_depth':depth, 'beta':beta, 'threshold':thresh, 'score':score}
                results.append(curr_res)
            print(f"best_thresh_score: {best_thresh_score}")
            fold_scores.append(best_thresh_score)
    
        # average across folds
        mean_score = np.mean(fold_scores)
        if mean_score > best_params['score']:
            best_params.update(
                max_depth=depth,
                beta=beta,
                threshold=best_thr,
                score=mean_score
            )
    
    results = pd.DataFrame(results)
    beta_to_plot = 1.0
    
    # Filter and pivot
    sub = results[results['beta'] == beta_to_plot]
    pivot = sub.pivot(index='max_depth', columns='threshold', values='score')
    
    # Plot heatmap
    matrix = pivot.values.astype(float)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect='auto', origin='lower')
    plt.colorbar(label=f'CV F$_{{{beta_to_plot}}}$ Score')
    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=[f"{t:.2f}" for t in pivot.columns],
        rotation=90
    )
    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=[str(d) for d in pivot.index]
    )
    plt.xlabel('Threshold')
    plt.ylabel('Max Depth')
    plt.title(f'CV F$_{{{beta_to_plot}}}$ Heatmap over Depth & Threshold')
    plt.tight_layout()
    plt.show()
    
    
    print("Best params: ",
          f"max_depth={best_params['max_depth']}, ",
          f"beta={best_params['beta']}, ",
          f"threshold={best_params['threshold']:.2f']} → ",
          f"CV F_{best_params['beta']}={best_params['score']}")
    
    
    for depth in [1,3,5,10, None]:
        rf = RandomForestClassifier(random_state=frozen_seed, verbose=2, n_jobs=-1, max_depth=depth) #class_weight='balanced')
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
    '''

if __name__ == '__main__':
    #main()




    # 379 first game
    file_path = "../full season data/plays_with_onehot_v1.csv"

    frozen_seed = 42

    freeze_seeds(frozen_seed)

    nba_dataset = get_dataset(file_path)

    train_dataset, test_dataset = train_test_split(nba_dataset, test_size=0.2, shuffle=True,
                                                   random_state=frozen_seed)
    X_train = train_dataset.drop(columns=['is_highlight'])
    y_train = train_dataset['is_highlight']
    X_test = test_dataset.drop(columns=['is_highlight'])
    y_test = test_dataset['is_highlight']
    max_depth=10
    rf = RandomForestClassifier(max_depth=10, random_state=frozen_seed)
    rf.fit(X_train, y_train)

    y_probs_train = rf.predict_proba(X_train)[:, 1]  # Probabilities for class 1
    y_probs_test = rf.predict_proba(X_test)[:, 1]
    # Define a custom threshold
    threshold = 0.07

    # Apply the threshold
    train_pred_threshold = (y_probs_train >= threshold).astype(int)
    test_pred_threshold = (y_probs_test >= threshold).astype(int)

    # Evaluate on training data
    print(f"On training with max_depth={max_depth}, threshold={threshold}:")
    print("Accuracy:", accuracy_score(y_train, train_pred_threshold))
    print("Classification Report:\n", classification_report(y_train, train_pred_threshold))
    print("Confusion Matrix:\n", confusion_matrix(y_train, train_pred_threshold))

    # Evaluate on test data
    print(f"On test with max_depth={max_depth}, threshold={threshold}:")
    print("Accuracy:", accuracy_score(y_test, test_pred_threshold))
    print("Classification Report:\n", classification_report(y_test, test_pred_threshold))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred_threshold))

    # sorted_probabilities = test_non_encode.sort_values(by='probability_class_1', ascending=False)
    # sorted_probabilities.to_csv('output_test_sorted_probabilities.csv', index=False)
    print("OK :)")



