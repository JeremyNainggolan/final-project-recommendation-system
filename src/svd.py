import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import random
import time  # Added for timing

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("Starting data loading and preprocessing...")
start_time = time.time()

# Load data
train_df = pd.read_csv('./data/train.csv')
test_users_df = pd.read_csv('./data/data_target_users_test.csv')
sample_submission_df = pd.read_csv('./data/sample_submission.csv')  # For reference

# Preprocessing: Prepare user interactions dict
user_interactions = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

# Compute global item popularity for baseline and cold-start fallback
item_popularity = train_df['item_id'].value_counts().sort_values(ascending=False)
top_10_items = item_popularity.head(10)

print(f"Data loading and preprocessing completed in {time.time() - start_time:.2f} seconds.")

# Function for baseline recommendations (item popularity)
def get_popularity_recommendations(user_id, top_items, user_interactions):
    interacted_items = set(user_interactions.get(user_id, []))
    recommendations = [item for item in top_items.index if item not in interacted_items][:10]
    return recommendations  # Return list for evaluation; join to string later

# Prepare sparse matrix for SVD
user_ids = train_df['user_id'].unique()
item_ids = train_df['item_id'].unique()
user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
idx_to_item_id = {idx: iid for iid, idx in item_id_to_idx.items()}

rows = [user_id_to_idx[uid] for uid in train_df['user_id']]
cols = [item_id_to_idx[iid] for iid in train_df['item_id']]
data = [1] * len(train_df)
user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))

print(f"User-item matrix created: {user_item_matrix.shape[0]} users, {user_item_matrix.shape[1]} items.")

# Train SVD model (tuned parameters; adjust as needed)
print("Starting SVD training on full data...")
svd_start = time.time()
svd = TruncatedSVD(n_components=128, random_state=42)
U = svd.fit_transform(user_item_matrix)  # User factors, shape (n_users, n_components)
V = svd.components_  # Item factors, shape (n_components, n_items)
S = svd.singular_values_  # Singular values, shape (n_components,)
print(f"SVD training completed in {time.time() - svd_start:.2f} seconds.")

# Function for SVD recommendations
def get_svd_recommendations(user_id, U, S, V, user_id_to_idx, idx_to_item_id, user_interactions, top_items):
    if user_id not in user_id_to_idx:
        # Cold-start: Fall back to popularity
        interacted_items = set(user_interactions.get(user_id, []))
        recommendations = [item for item in top_items.index if item not in interacted_items][:10]
        return recommendations
    
    user_idx = user_id_to_idx[user_id]
    user_vec = U[user_idx]  # shape (n_components,)
    scores = user_vec @ (np.diag(S) @ V)  # shape (n_items,)
    
    # Mask out interacted items
    interacted_items = user_interactions.get(user_id, [])
    interacted_indices = [item_id_to_idx[iid] for iid in interacted_items if iid in item_id_to_idx]
    scores[interacted_indices] = -np.inf
    
    # Get top 10 recommendations
    top_indices = np.argsort(scores)[::-1][:10]
    recommendations = [idx_to_item_id[idx] for idx in top_indices]
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
print("Creating validation set...")
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

print(f"Validation set created for {len(validation_ground_truth)} users.")

# Rebuild user_item_matrix with train_interactions only (for fair evaluation)
train_df_filtered = []
for user_id, items in train_interactions.items():
    for item in items:
        train_df_filtered.append({'user_id': user_id, 'item_id': item})
train_df_filtered = pd.DataFrame(train_df_filtered)

rows_train = [user_id_to_idx[uid] for uid in train_df_filtered['user_id']]
cols_train = [item_id_to_idx[iid] for iid in train_df_filtered['item_id']]
data_train = [1] * len(train_df_filtered)
user_item_matrix_train = csr_matrix((data_train, (rows_train, cols_train)), shape=(len(user_ids), len(item_ids)))

# Retrain SVD on filtered train data
print("Retraining SVD on filtered train data...")
svd_train_start = time.time()
U_train = svd.fit_transform(user_item_matrix_train)
V_train = svd.components_
S_train = svd.singular_values_
print(f"SVD retraining completed in {time.time() - svd_train_start:.2f} seconds.")

# Generate recommendations for validation users
print("Generating recommendations for validation users...")
rec_start = time.time()
baseline_recommendations = {}
svd_recommendations = {}
total_users = len(validation_ground_truth)
for i, user_id in enumerate(validation_ground_truth.keys()):
    baseline_recommendations[user_id] = get_popularity_recommendations(user_id, top_10_items, train_interactions)
    svd_recommendations[user_id] = get_svd_recommendations(user_id, U_train, S_train, V_train, user_id_to_idx, idx_to_item_id, train_interactions, top_10_items)
    if (i + 1) % 100 == 0:  # Print progress every 100 users
        print(f"Processed {i + 1}/{total_users} validation users...")
print(f"Recommendation generation completed in {time.time() - rec_start:.2f} seconds.")

# Evaluate on validation set
print("Evaluating on validation set...")
eval_start = time.time()
baseline_map10 = mean_average_precision_at_k(baseline_recommendations, validation_ground_truth, k=10)
svd_map10 = mean_average_precision_at_k(svd_recommendations, validation_ground_truth, k=10)
print(f"Evaluation completed in {time.time() - eval_start:.2f} seconds.")
print(f"Baseline MAP@10: {baseline_map10:.4f}")
print(f"SVD MAP@10: {svd_map10:.4f}")

# Generate full recommendations for test users (using original train data)
# Retrain SVD on full train data for final submission
print("Retraining SVD on full data for final submission...")
final_svd_start = time.time()
U_full = svd.fit_transform(user_item_matrix)
V_full = svd.components_
S_full = svd.singular_values_
print(f"Final SVD retraining completed in {time.time() - final_svd_start:.2f} seconds.")

print("Generating final recommendations for test users...")
final_rec_start = time.time()
baseline_submission = []
svd_submission = []
total_test_users = len(test_users_df)
for i, user_id in enumerate(test_users_df['user_id']):
    baseline_recs = get_popularity_recommendations(user_id, top_10_items, user_interactions)
    svd_recs = get_svd_recommendations(user_id, U_full, S_full, V_full, user_id_to_idx, idx_to_item_id, user_interactions, top_10_items)
    
    baseline_submission.append({'user_id': user_id, 'item_id': ' '.join(baseline_recs)})
    svd_submission.append({'user_id': user_id, 'item_id': ' '.join(svd_recs)})
    
    if (i + 1) % 100 == 0:  # Print progress every 100 users
        print(f"Processed {i + 1}/{total_test_users} test users...")
print(f"Final recommendation generation completed in {time.time() - final_rec_start:.2f} seconds.")

# Save submissions
print("Saving submissions...")
baseline_df = pd.DataFrame(baseline_submission)
baseline_df.to_csv('/baseline_submission.csv', index=False)

svd_df = pd.DataFrame(svd_submission)
svd_df.to_csv('/svd_submission.csv', index=False)

print("Submissions saved: /baseline_submission.csv and /svd_submission.csv")
print(f"Total script execution time: {time.time() - start_time:.2f} seconds.")
