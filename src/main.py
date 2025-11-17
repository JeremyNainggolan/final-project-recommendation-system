import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
train_df = pd.read_csv('./data/train.csv')
test_users_df = pd.read_csv('./data/data_target_users_test.csv')
sample_submission_df = pd.read_csv('./data/sample_submission.csv')  # For reference

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
        train_df_filtered.append({'user_id': user_id, 'item_id': item})
train_df_filtered = pd.DataFrame(train_df_filtered)

rows_train = [user_id_to_idx[uid] for uid in train_df_filtered['user_id']]
cols_train = [item_id_to_idx[iid] for iid in train_df_filtered['item_id']]
data_train = [1] * len(train_df_filtered)
user_item_matrix_train = csr_matrix((data_train, (rows_train, cols_train)), shape=(len(user_ids), len(item_ids)))

# Hyperparameter tuning for ALS
# Define parameter grids (reduced for efficiency; expand if needed)
factors_list = [64, 128, 256]
regularization_list = [0.001, 0.01, 0.1]
iterations_list = [10, 20, 30]
alpha_list = [20, 40, 80]

best_params = None
best_score = 0.0

print("Starting hyperparameter tuning...")
for factors in factors_list:
    for reg in regularization_list:
        for iters in iterations_list:
            for alpha in alpha_list:
                print(f"Testing: factors={factors}, reg={reg}, iters={iters}, alpha={alpha}")
                model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, alpha=alpha, random_state=42)
                model.fit(user_item_matrix_train)
                
                # Generate recommendations for validation users
                als_recommendations = {}
                for user_id in validation_ground_truth.keys():
                    als_recommendations[user_id] = get_als_recommendations(user_id, model, user_item_matrix_train, user_id_to_idx, idx_to_item_id, train_interactions, top_10_items)
                
                # Evaluate
                als_map10 = mean_average_precision_at_k(als_recommendations, validation_ground_truth, k=10)
                print(f"MAP@10: {als_map10:.4f}")
                
                if als_map10 > best_score:
                    best_score = als_map10
                    best_params = {'factors': factors, 'regularization': reg, 'iterations': iters, 'alpha': alpha}

print(f"Best params: {best_params}, Best MAP@10: {best_score:.4f}")

# Use best params for final model
model = AlternatingLeastSquares(**best_params, random_state=42)
model.fit(user_item_matrix_train)

# Evaluate best model on validation
als_recommendations = {}
for user_id in validation_ground_truth.keys():
    als_recommendations[user_id] = get_als_recommendations(user_id, model, user_item_matrix_train, user_id_to_idx, idx_to_item_id, train_interactions, top_10_items)

als_map10 = mean_average_precision_at_k(als_recommendations, validation_ground_truth, k=10)

# Baseline for comparison
baseline_recommendations = {}
for user_id in validation_ground_truth.keys():
    baseline_recommendations[user_id] = get_popularity_recommendations(user_id, top_10_items, train_interactions)

baseline_map10 = mean_average_precision_at_k(baseline_recommendations, validation_ground_truth, k=10)

print(f"Baseline MAP@10: {baseline_map10:.4f}")
print(f"Optimized ALS MAP@10: {als_map10:.4f}")

# Generate full recommendations for test users (using original train data)
# Retrain ALS on full train data for final submission
model.fit(user_item_matrix)

als_submission = []
for user_id in test_users_df['user_id']:
    als_recs = get_als_recommendations(user_id, model, user_item_matrix, user_id_to_idx, idx_to_item_id, user_interactions, top_10_items)
    als_submission.append({'user_id': user_id, 'item_id': ' '.join(map(str, als_recs))})

# Save submission
als_df = pd.DataFrame(als_submission)
als_df.to_csv('/optimized_als_submission.csv', index=False)

print("Optimized ALS submission saved: /optimized_als_submission.csv")
