"""
Group Recommender System based on Members' Influence and Leader Impact

This code implements a group recommender system inspired by 
the paper "A novel group recommender system based on members’ influence and leader impact" by 
Reza Barzegar Nozari and Hamidreza Koohi. 
The system calculates influenced ratings for group members and evaluates recommendations using various metrics. 
It aims to suggest items to a group of people who share common interests while considering social relationships and 
the influence of individuals on each other in the group.

Paper Reference:
Reza Barzegar Nozari and Hamidreza Koohi. (2020).
A novel group recommender system based on members’ influence and leader impact.
Knowledge-Based Systems, 205, 106296. https://doi.org/10.1016/j.knosys.2020.106296

The code is organized as follows:

1. Import Statements: Import necessary libraries for data manipulation and analysis.

2. Function Definitions:
   - calculate_trust: Calculate the trust matrix based on similarity and common rated items.
   - calculate_similarity: Calculate Pearson correlation coefficient similarity between members.
   - identify_leader: Identify the leader within a group based on Trust and Similarity matrices.
   - calculate_influence_weight: Calculate influence weight based on leader's impact, similarity, and trust.
   - influenced_rating: Calculate influenced ratings for group members and items.
   - evaluate_recommendations: Evaluate recommendations using various metrics.

3. Main Function:
   - Read group ratings from a CSV file.
   - Calculate influenced ratings.
   - Determine group rating for items using an averaging method.
   - Evaluate the recommendations.
   - Print and save the evaluation results to an Excel file.

"""

# Impoting necessary libraries
import numpy as np
import pandas as pd
from scipy.spatial import distance


# Function to calculate trust matrix based on common rated items and ratings distance
def calculate_trust(Group):
    """
    Calculate the trust matrix based on common rated items between group members and their ratings distance.

    Parameters:
        Group (DataFrame): A DataFrame containing group members' ratings for items.

    Returns:
        Trust_matrix (DataFrame): The trust matrix representing the trust levels between members.
    """
    members = Group.index
    no_member = len(members)
    
    Trust_matrix = pd.DataFrame(0.0, index=members, columns=members)
    
    for u in members:
        rated_list_u = Group.loc[u].index[Group.loc[u] > 0]
        count_rated_u = len(rated_list_u)
        ratings_u = Group.loc[u][:]
        
        for v in members:
            if u == v:
                continue
            
            rated_list_v = Group.loc[v].index[Group.loc[v] > 0]
            count_rated_v = len(rated_list_v)
            ratings_v = Group.loc[v][:]
            
            intersection_uv = set(rated_list_u).intersection(rated_list_v)
            count_intersection = len(intersection_uv)
            
            partnership_uv = count_intersection / count_rated_u
            
            dst_uv = 1 / (1 + distance.euclidean(ratings_u, ratings_v))
            
            trust_uv = (2 * partnership_uv * dst_uv) / (partnership_uv + dst_uv) 
            Trust_matrix.at[u, v] = trust_uv
            
    return Trust_matrix


# Function to calculate Pearson correlation coefficient similarity between members
def calculate_similarity(Group):
    """
    Calculate the Pearson correlation coefficient (PCC) similarity matrix between group members.

    Parameters:
        Group (DataFrame): A DataFrame containing group members' ratings for items.

    Returns:
        PCC_df (DataFrame): The PCC similarity between group members.
    """
    members = Group.index
    ratings = Group.to_numpy()  # Convert DataFrame to a NumPy array

    # Calculate the Pearson correlation coefficient similarity
    PCC = np.corrcoef(ratings, rowvar=True)
    
    # Convert the matrix to a DataFrame with proper index and columns
    PCC_df = pd.DataFrame(PCC, index=members, columns=members)

    return PCC_df


# Function to identify leader within a group based on Trust and Similarity matrices
def identify_leader(Trust_matrix, Similarity_matrix, total_members):
    """
    Identify the leader within a group based on Trust and Similarity matrices.

    Parameters:
        Trust_matrix (DataFrame): The trust matrix representing the trust levels between members.
        Similarity_matrix (DataFrame): The PCC similarity matrix between group members.
        total_members (int): Total number of members in the group.

    Returns:
        leader_id (int): ID of the identified leader.
        leader_impact (float): Impact value of the identified leader on group preferences.
    """
    trust_sum = np.sum(Trust_matrix.values, axis=0) - 1
    similarity_sum = np.sum(Similarity_matrix.values, axis=0) - 1
    ts_sumation = trust_sum + similarity_sum

    LeaderId = np.argmax(ts_sumation)
    LeaderImpact = ts_sumation[LeaderId] / (total_members - 1)

    return Trust_matrix.index[LeaderId], LeaderImpact


# Function to calculate influence weight based on leader's impact, similarity, and trust
def calculate_influence_weight(leader_id, leader_impact, similarity_uv, trust_uv, v):
    """
    Calculate the influence weight based on leader's impact, similarity, and trust.

    Parameters:
        leader_id (int): ID of the identified leader.
        leader_impact (float): Impact value of the identified leader on group preferences.
        similarity_uv (float): Similarity score between two members.
        trust_uv (float): Trust score between two members.
        v (int): ID of the member being considered.

    Returns:
        weight_uv (float): Calculated influence weight.
    """    
    if v == leader_id:
        weight_uv = (1/2) * ((leader_impact + (similarity_uv * trust_uv)) / (similarity_uv + trust_uv))
    else:
        weight_uv = (similarity_uv * trust_uv) / (similarity_uv + trust_uv)
        
    return weight_uv


# Function to calculate influenced ratings for group members and items
def influenced_rating(group):
    """
    Calculate influenced ratings for group members and items.

    Parameters:
        group (DataFrame): A DataFrame containing group members' ratings for items.

    Returns:
        influenced_ratings (DataFrame): DataFrame containing influenced ratings for group members and items.
    """    
    members = group.index
    movies = group.columns
    num_members, num_items = len(members), len(movies)

    # Calculate trust and similarity matrices
    trust_matrix = calculate_trust(group)
    similarity_matrix = calculate_similarity(group)

    # Identify the leader and their impact
    leader_id, leader_impact = identify_leader(trust_matrix, similarity_matrix, num_members)

    influenced_ratings = pd.DataFrame(0.0, index=members, columns=movies)

    for u in members:
        for i in movies:
            score_ui = group.at[u, i]
            influence = 0

            if score_ui > 0:
                for v in members:
                    if v != u:
                        score_vi = group.at[v, i]
                        similarity_uv = similarity_matrix.at[u, v]
                        trust_uv = trust_matrix.at[u, v]
                        weight_vu = calculate_influence_weight(leader_id, leader_impact, similarity_uv, trust_uv, v)

                        if score_vi > 0:
                            influence += weight_vu * (score_vi - score_ui)

                influenced_ratings.at[u, i] = score_ui + influence

    return influenced_ratings


# Function to evaluate recommendations using metrics
def evaluate_recommendations(Group, Group_Rating, rec_size, satisfied_Tr):
    """
    Evaluate recommendations based on various metrics.

    Parameters:
        Group (DataFrame): A DataFrame containing group members' ratings for items.
        aggregation (Series): Aggregated ratings for items.
        rec_size (int): Number of recommendations.
        satisfied_Tr (float): Threshold for satisfaction.

    Returns:
        results (dict): Dictionary containing evaluation results.
    """    
    Group_Rating = Group_Rating.sort_values(ascending=False)
    rec_list = Group_Rating[Group_Rating != 0]
    
    recommendation_index = rec_list.index
    members = Group.index
    no_member = len(members)
    
    TP = TN = FP = FN = 0
    satisfied = 0
    
    for r, index in enumerate(recommendation_index):
        for u in members:
            preference_u_ind = Group.at[u, index]
            
            if r < rec_size:
                if preference_u_ind >= satisfied_Tr:
                    satisfied += 1
                    TP += 1
                else:
                    FP += 1
            else:
                if preference_u_ind >= satisfied_Tr:
                    FN += 1
                else:
                    TN += 1
                    
    total_count = TP + FP + TN + FN
    
    accuracy = ((TP + TN) / total_count) * 100 if total_count > 0 else 0
    precision = (TP / (TP + FP)) * 100 if TP + FP > 0 else 0
    recall = (TP / (TP + FN)) * 100 if TP + FN > 0 else 0
    specificity = (TN / (TN + FP)) * 100 if TN + FP > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    balanced_accuracy = (specificity + recall) / 2
    
    results = {
        "Satisfaction": satisfied / (no_member * rec_size),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Balanced_Accuracy": balanced_accuracy,
        "F1_Score": f1_score,
        "Confusion_counters": {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
    }
    
    return results


# Main function to execute the recommendation system
def main():
    """
    Main function to execute the group recommendation system.

    Reads group ratings from a CSV file, calculates influenced ratings, evaluates recommendations,
    and prints the evaluation results.
    """    
    Group = pd.read_csv('Groups/Group_q2.csv')

    users_id = Group["Unnamed: 0"].unique()
    Group = Group.drop(['Unnamed: 0'], axis=1)
    Group = Group.set_axis(users_id, axis='rows')

    # Calculate members' influenced ratings
    Influenced_Ratings = influenced_rating(Group)

    # Determine group rating for items using averaging aggregation method
    Group_Rating = Influenced_Ratings.mean(axis=0).fillna(0)

    # Evaluate the recommendations
    rec_size = 5
    satisfied_Tr = 4
    Evaluation_Results = evaluate_recommendations(Group, Group_Rating, rec_size, satisfied_Tr)

    print("Evaluation Results:", Evaluation_Results)

    # Save the evaluation results to an Excel file
    results_df = pd.DataFrame(Evaluation_Results, index=["Evaluation Results"])
    results_df.to_excel('Results/Evaluation_Results.xlsx')


if __name__ == "__main__":
    main()
