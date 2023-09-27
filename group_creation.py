'''
# Group Creation based on "A Novel Group Recommender System" by Reza Barzegar Nozari and Hamidreza Koohi

This Python script implements a group creation method inspired by the paper "A Novel Group Recommender System Based 
on Members’ Influence and Leader Impact" by Reza Barzegar Nozari and Hamidreza Koohi. The method uses 
Fuzzy C-Means Clustering and Pearson Correlation Coefficient (PCC) to select a group of similar users with 
shared preferences for items in a user-item dataset.

## Features

- Reads a user-item dataset in CSV format.
- Creates a user-item matrix to represent user-item interactions.
- Normalizes the user-item matrix using Min-Max scaling.
- Performs Fuzzy C-Means Clustering on the normalized matrix to group users.
- Filters users by a specified cluster label.
- Calculates PCC between a randomly selected user and other users in the same cluster.
- Selects the top similar users based on PCC values.
- Outputs and saves the selected group of similar users to a CSV file.


## Parameters

- dataset_path: Path to your user-item dataset in CSV format.
- target_cluster: The cluster label from which to select similar users.
- group_size: The number of similar users to select for the group.
- c: Number of clusters for Fuzzy C-Means Clustering (default is 3).
- m: Fuzzy exponent for FCM clustering (default is 80).
- max_iter: Maximum number of iterations for FCM clustering (default is 1000).

Feel free to adjust these parameters based on your dataset and requirements.
'''
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from scipy.stats import pearsonr

def read_dataset(dataset_path):
    # Read the dataset from the given path
    return pd.read_csv(dataset_path)

def create_user_item_matrix(df):
    # Extract unique user and movie IDs
    movies_id = np.sort(df["movieId"].unique())
    users_id = np.sort(df["userId"].unique())

    # Create the user-item matrix (UIM)
    UIM = pd.DataFrame(np.zeros((len(users_id), len(movies_id))))
    UIM = UIM.set_axis(movies_id, axis='columns')
    UIM = UIM.set_axis(users_id, axis='rows')

    # Fill the user-item matrix with ratings
    for index, row in df.iterrows():
        u = row['userId']
        m = row['movieId']
        s = row['rating']
        UIM.at[u, m] = s

    return UIM

def normalize_user_item_matrix(UIM):
    # Normalize the matrix using Min-Max scaling
    min_rating = np.min(UIM)
    max_rating = np.max(UIM)
    return (UIM - min_rating) / (max_rating - min_rating)

def perform_fuzzy_cmeans_clustering(normalized_matrix, c, m, max_iter):
    # FCM clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=normalized_matrix.T, c=c, m=m, error=0.01, maxiter=max_iter, init=None
    )

    # Assign cluster labels to users
    cluster_membership = np.argmax(u, axis=0)
    return cluster_membership

def filter_users_by_cluster(UIM, cluster_membership, target_cluster):
    # Combine the user-item matrix with cluster labels
    cluster_labaled_UIM = pd.concat([UIM, pd.DataFrame(cluster_membership, index=UIM.index, columns=["Cluster Label"])], axis=1)

    # Filter users from the target cluster
    cluster_data = cluster_labaled_UIM[cluster_labaled_UIM["Cluster Label"] == target_cluster]

    # Delete the "Cluster Label" column
    del cluster_data["Cluster Label"]

    return cluster_data

def calculate_pcc(user1_ratings, user2_ratings):
    # Check if both users have rated at least one common item
    common_items = np.intersect1d(user1_ratings.columns, user2_ratings.columns)
    if len(common_items) > 0:
        return pearsonr(np.array(user1_ratings)[0], np.array(user2_ratings)[0])[0]
    return None

def select_top_similar_users(cluster_data, group_size):
    # Select a random user from the cluster
    random_user = np.random.choice(cluster_data.index.unique())

    # Calculate PCC with the randomly selected user for all users in the same cluster
    pcc_values = []

    for user_id in cluster_data.index.unique():
        if user_id != random_user:
            user1_ratings = cluster_data[cluster_data.index == random_user]
            user2_ratings = cluster_data[cluster_data.index == user_id]

            pcc = calculate_pcc(user1_ratings, user2_ratings)
            if pcc is not None:
                pcc_values.append((user_id, pcc))

    # Sort users by PCC values in descending order
    pcc_values.sort(key=lambda x: x[1], reverse=True)

    # Extract the user IDs of the top similar users
    similar_user_ids = [user_id for user_id, _ in pcc_values][:group_size]

    # Create a new DataFrame containing data of the top similar users as a group
    group_data = cluster_data[cluster_data.index.isin(similar_user_ids)]

    return group_data


# Main function to orchestrate the entire process
def main(dataset_path, target_cluster, group_size, c=3, m=80, max_iter=1000):
    # Step 1: Read the dataset
    df = read_dataset(dataset_path)

    # Step 2: Create the user-item matrix
    UIM = create_user_item_matrix(df)

    # Step 3: Normalize the user-item matrix
    normalized_matrix = normalize_user_item_matrix(UIM)

    # Step 4: Perform Fuzzy C-Means Clustering
    cluster_membership = perform_fuzzy_cmeans_clustering(normalized_matrix, c, m, max_iter)

    # Step 5: Filter users by cluster
    cluster_data = filter_users_by_cluster(UIM, cluster_membership, target_cluster)

    # Step 6: Select top similar users as a group
    similar_users_group = select_top_similar_users(cluster_data, group_size)
    
    print(similar_users_group)
    
    similar_users_group.to_csv('Groups/Group_data.csv')


# Example usage:
if __name__ == "__main__":
    
    dataset_path = 'data/Data.csv'
    target_cluster = 1
    group_size = 5

    main(dataset_path, target_cluster, group_size)
    