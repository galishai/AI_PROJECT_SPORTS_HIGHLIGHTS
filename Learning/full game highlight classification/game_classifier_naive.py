from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import warnings

from Learning.dataset_helper_functions import *
from Learning.MLPClassifier import *
from sklearn.metrics import classification_report

warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns.*",
    category=DeprecationWarning,
)


HIGHLIHTS_MEAN_1Q = 4
HIGHLIHTS_MEAN_2Q = 5
HIGHLIHTS_MEAN_3Q = 5
HIGHLIHTS_MEAN_4Q = 5

INC_FTS = True

INC_TIES = True

highlights_per_qtr = {'1st': HIGHLIHTS_MEAN_1Q, '2nd': HIGHLIHTS_MEAN_2Q, '3rd':HIGHLIHTS_MEAN_3Q, '4th':HIGHLIHTS_MEAN_4Q}

seed = 42
data_path = "../../full season data/plays_with_onehot_v2.csv"

RM_FT_MODEL = False
#pd.set_option('display.max_rows', None, 'display.max_columns', None)

trained_model_params={
    'hidden_dim' : 128,
    'dropout' : 0.3,
}

def select_highlights_qtr(g, n):
    top_n =  g[~g['play'].isin(free_throw_play_ids)].nlargest(n, 'prob')
    tied_times = top_n['time_left_qtr'].unique()
    return g[g['time_left_qtr'].isin(tied_times)]


def main():
    #data prep
    freeze_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print('Using device:', device)

    unaltered_df = pd.read_csv(data_path)
    df = get_dataset(path=data_path, verbose=False, rm_ft_ds=False, add_game_idx=True)

    start_positions = list((
        df.groupby("game_id")
        .apply(lambda g: g.index[0])
        .values
    ))
    end_positions = start_positions[1:] + [len(df)]

    X = df.drop(columns=['is_highlight', 'game_id'])

    if RM_FT_MODEL:
        save_path = "../saved_model/mlp_final_checkpoint_rm_ft.pth"
    else:
        save_path = "../saved_model/mlp_final_checkpoint.pth"
    checkpoint = torch.load(save_path, map_location=device)
    model = MLPClassifier(input_dim=X.shape[1], hidden_dim=trained_model_params['hidden_dim'],
                          dropout=trained_model_params['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    pred_accuracies = []
    all_preds = []
    all_labels = []
    all_probs = []
    for i, (start, end) in enumerate(zip(start_positions, end_positions)):
        df_game = df.iloc[start:end].copy()
        df_orig = unaltered_df.iloc[start:end].copy()

        X_game = df_game.drop(columns=['is_highlight', 'game_id']).values.astype(np.float32)
        X_tensor = torch.from_numpy(X_game).to(device)

        logits = model(X_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        df_game['prob'] = probs
        df_game['play_id'] = df_orig['play']
        quarters = ['1st','2nd','3rd','4th']
        df_game['predicted'] = 0
        for q in quarters:
            mask_q = df_game[f'quarter_{q}'] == 1
            #print(f"plays in {q}: {sum(mask_q)}")
            n = highlights_per_qtr.get(q,0)
            if INC_FTS:
                selected = (df_game[mask_q].nlargest(n,'prob'))
            else:
                selected = (df_game[mask_q & ~df_game['play_id'].isin(free_throw_play_ids)].nlargest(n, 'prob'))
            if INC_TIES:
                tied_times = selected['time_left_qtr'].unique()
                #print(f"quarter: {q}, select len: {sum(mask_q & df_game['time_left_qtr'].isin(tied_times))}")
                df_game.loc[mask_q & df_game['time_left_qtr'].isin(tied_times), 'predicted'] = 1
            else:
                df_game.loc[selected.index, 'predicted'] = 1
        y_pred = df_game['predicted']
        y_truth = df_game['is_highlight']
        accuracy = (y_truth == y_pred).mean()
        pred_accuracies.append(accuracy)
        print(f"Game prediction accuracy: {accuracy:.3%}")
        all_preds.extend(y_pred)
        all_labels.extend(y_truth)
        #exit(0)

    print(f"Game prediction Final Mean Accuracy: {(sum(pred_accuracies)/len(pred_accuracies)):.3%}")
    print(classification_report(all_labels, all_preds, digits=3))
    unaltered_df['predicted'] = all_preds
    unaltered_df['probs'] = all_probs
    #unaltered_df.to_csv("predicted_output.csv", index=False)


if __name__ == '__main__':
    main()