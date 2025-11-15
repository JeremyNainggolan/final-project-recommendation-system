import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
train_df = pd.read_csv('train.csv')
test_users_df = pd.read_csv('data_target_users_test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')  # For reference

# Preprocessing: Prepare user interactions dict
user_interactions = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

# Compute global item popularity for baseline and cold-start fallback
item_popularity = train_df['item_id'].value_counts().sort_values(ascending=False)
top_10_items = item_popularity.head(10)

# Function for baseline recommendations (item popularity)
def get_popularity_recommendations(user_id, top_items, user_interactions):
    interacted_items = set(user_interactions.get(user_id, []))
    recommendations = [item for item in top_items.index if item not in interacted_items][:10]
    return recommendations  # Return list for evaluation; join to string later

# Prepare sparse matrix for ALS
user_ids = train_df['user_id'].unique()
item_ids = train_df['item_id'].unique()
user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
idx_to_item_id = {idx: iid for iid, idx in item_id_to_idx.items()}

rows = [user_id_to_idx[uid] for uid in train_df['user_id']]
cols = [item_id_to_idx[iid] for iid in train_df['item_id']]
data = [1] * len(train_df)
user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))

# --- PERUBAHAN DI SINI (SET 3) ---
# Train ALS model (tuned parameters; Set 3)
model_params = {
    'factors': 64,         # Lebih rendah untuk generalisasi
    'regularization': 0.05,  # Nilai regularisasi yang berbeda
    'iterations': 25,      # Iterasi yang cukup
    'alpha': 25            # Nilai confidence yang berbeda
}
model = AlternatingLeastSquares(**model_params)
# ---------------------------------
model.fit(user_item_matrix)

# Function for ALS recommendations
def get_als_recommendations(user_id, model, user_item_matrix, user_id_to_idx, idx_to_item_id, user_interactions, top_items):
    if user_id not in user_id_to_idx:
        # Cold-start: Fall back to popularity
        interacted_items = set(user_interactions.get(user_id, []))
        recommendations = [item for item in top_items.index if item not in interacted_items][:10]
        return recommendations
    
    user_idx = user_id_to_idx[user_id]
    recs_indices, _ = model.recommend(user_idx, user_item_matrix[user_idx], N=10)
    recommendations = [idx_to_item_id[idx] for idx in recs_indices]
    return recommendations

# Evaluation functions
def average_precision_at_k(recommended, relevant, k=10):
    if not recommended or not relevant:
        return 0.0
    
    ap = 0.0
    num_relevant = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            ap += precision_at_i
    
    num_relevant_in_top_k = sum(1 for item in recommended[:k] if item in relevant)
    if num_relevant_in_top_k == 0:
        return 0.0
    return ap / num_relevant_in_top_k

def mean_average_precision_at_k(recommendations, ground_truth, k=10):
    ap_scores = []
    for user_id, recs in recommendations.items():
        relevant = ground_truth.get(user_id, set())
        ap = average_precision_at_k(recs, relevant, k)
        ap_scores.append(ap)
    
    if not ap_scores:
        return 0.0
    return sum(ap_scores) / len(ap_scores)

# Create validation set (80/20 split)
validation_ground_truth = {}
train_interactions = {}
for user_id, items in user_interactions.items():
    if len(items) > 1:  # Ensure at least one for train and one for validation
        random.shuffle(items)
        split_idx = int(0.8 * len(items))
        train_interactions[user_id] = items[:split_idx]
        validation_ground_truth[user_id] = set(items[split_idx:])
    else:
        train_interactions[user_id] = items
        validation_ground_truth[user_id] = set()

# Rebuild user_item_matrix with train_interactions only (for fair evaluation)
train_df_filtered = []
for user_id, items in train_interactions.items():
    for item in items:
        if item in item_id_to_idx:
            train_df_filtered.append({'user_id': user_id, 'item_id': item})
train_df_filtered = pd.DataFrame(train_df_filtered)

rows_train = [user_id_to_idx[uid] for uid in train_df_filtered['user_id']]
cols_train = [item_id_to_idx[iid] for iid in train_df_filtered['item_id']]
data_train = [1] * len(train_df_filtered)
user_item_matrix_train = csr_matrix((data_train, (rows_train, cols_train)), shape=(len(user_ids), len(item_ids)))

# Retrain ALS on filtered train data
# --- PERUBAHAN DI SINI (SET 3) ---
model_eval = AlternatingLeastSquares(**model_params)
# ---------------------------------
model_eval.fit(user_item_matrix_train)

# Generate recommendations for validation users
baseline_recommendations = {}
als_recommendations = {}
for user_id in validation_ground_truth.keys():
    baseline_recommendations[user_id] = get_popularity_recommendations(user_id, top_10_items, train_interactions)
    als_recommendations[user_id] = get_als_recommendations(user_id, model_eval, user_item_matrix_train, user_id_to_idx, idx_to_item_id, train_interactions, top_10_items)

# Evaluate on validation set
baseline_map10 = mean_average_precision_at_k(baseline_recommendations, validation_ground_truth, k=10)
als_map10 = mean_average_precision_at_k(als_recommendations, validation_ground_truth, k=10)

print(f"Baseline MAP@10 (Validation): {baseline_map10:.4f}")
print(f"ALS (Set 3) MAP@10 (Validation): {als_map10:.4f}")

# Generate full recommendations for test users (using original train data)
# (We already trained 'model' on the full user_item_matrix at the beginning)

baseline_submission = []
als_submission = []
for user_id in test_users_df['user_id']:
    baseline_recs = get_popularity_recommendations(user_id, top_10_items, user_interactions)
    # Use the 'model' trained on the full dataset
    als_recs = get_als_recommendations(user_id, model, user_item_matrix, user_id_to_idx, idx_to_item_id, user_interactions, top_10_items)
    
    baseline_submission.append({'user_id': user_id, 'item_id': ' '.join(baseline_recs)})
    als_submission.append({'user_id': user_id, 'item_id': ' '.join(als_recs)})

# Save submissions
baseline_df = pd.DataFrame(baseline_submission)
baseline_df.to_csv('./baseline_submission.csv', index=False)

als_df = pd.DataFrame(als_submission)
als_df.to_csv('./als_submission_set3.csv', index=False) # Ganti nama file

print("Submissions saved: ./baseline_submission.csv and ./als_submission_set3.csv")