# IBGR: Influence-Based Group Recommendation Model

This repository contains a Python script that embodies the innovative approach detailed in the research paper "A novel group recommender system based on members’ influence and leader impact" by Reza Barzegar Nozari and Hamidreza Koohi. The paper addresses the intricacies of group recommender systems, which, unlike conventional systems, recommend items simultaneously to a group of individuals sharing common interests, with the aim of satisfying each member. A pivotal challenge in these systems lies in understanding social dynamics and the impact of individual preferences on the group's overall choices. The aim of this project is to explore and practically apply the concepts presented in the paper to build a recommendation system that caters to the preferences and dynamics within a group.

## Paper Overview

In the paper, the authors propose an innovative approach to group recommender systems, focusing on group creation and the influence of individuals on each other within a group. They introduce the concept of "Leaders" who have a significant impact and influence on the preferences of other group members. The paper discusses the use of trust, similarity, and leadership impact to enhance group recommendation outcomes. Additionally, the proposed method is evaluated using real-world data to demonstrate its effectiveness.

## Scripts Functionality

The scripts `group_creation.py` and `IBGR.py` translate the paper's theoretical framework into a functional system, creating groups and building a recommendation mechanism while meticulously evaluating its performance using a diverse array of metrics. Let's dissect the scripts' components in relation to the paper's conceptual foundations:

### Group Creation (group_creation.py)

1. **read_dataset Function:** 
   - Reads the user-item dataset in CSV format.

2. **create_user_item_matrix Function:** 
   - Creates a user-item matrix to represent user-item interactions.

3. **normalize_user_item_matrix Function:** 
   - Normalizes the user-item matrix using Min-Max scaling.

4. **perform_fuzzy_cmeans_clustering Function:** 
   - Performs Fuzzy C-Means Clustering on the normalized matrix to group users.

5. **filter_users_by_cluster Function:** 
   - Filters users by a specified cluster label.

6. **calculate_pcc Function:** 
   - Calculates Pearson Correlation Coefficient (PCC) between users in the same cluster.

7. **select_top_similar_users Function:** 
   - Selects a group of top similar users based on PCC values.

8. **main Function:** 
   - Orchestrates the entire process of group creation, starting from reading the dataset to selecting and saving the group data.


### Influence-Based Recommendation (IBGR.py)

1. **Import Statements:**
   - The script starts by importing essential libraries like `numpy` and `pandas` for data handling and analysis, along with `scipy.spatial.distance` for distance calculations, mirroring the methodological toolkit employed in the paper.

2. **`calculate_trust` Function:**
   - This function operationalizes the paper's proposition to compute trust matrices, which gauge the trustworthiness of group members. It mirrors the research's focus on assessing trust based on common rated items and ratings.

3. **`calculate_similarity` Function:**
   - The function aligns with the paper's conceptualization of computing similarity using Pearson correlation coefficients. It quantifies the resemblance in ratings among group members, a pivotal aspect of the proposed method.

4. **`identify_leader` Function:**
   - The function translates the paper's concept of identifying influential leaders in the group. By amalgamating Trust and Similarity matrices, the script quantifies leader impact in accordance with the paper's perspective.

5. **`calculate_influence_weight` Function:**
   - This function quantifies the influence weight for adjusting ratings, mirroring the paper's emphasis on considering the leader's role when determining the weight.

6. **`influenced_rating` Function:**
   - The function follows the paper's approach of computing influenced ratings. It captures how the ratings of one member are influenced by others, echoing the research's focus on trust, similarity, and leadership dynamics.

7. **`evaluate_recommendations` Function:**
   - The function's metrics-based evaluation replicates the paper's methodology. By assessing accuracy, precision, recall, and other metrics, it gauges recommendation efficacy, aligning with the paper's comprehensive evaluation approach.

8. **`main` Function:**
   - The central `main` function orchestrates the script's execution. It mirrors the paper's approach by reading group ratings, computing influenced ratings, evaluating recommendations, and presenting results.

9. **`if __name__ == "__main__":` Block:**
   - This segment mirrors the paper's approach to ensure the `main` function executes autonomously, aligning with the script's role as a standalone application.

In summary, these scripts operationalize the innovative method presented in the paper. They implement the proposed group recommendation system, create groups, compute influenced ratings, rigorously evaluate recommendation outcomes, and echoe the paper's findings by presenting comprehensive evaluation results.

## How to Use

1. Clone this repository to your local machine.
2. Create [Virtualenv](https://virtualenv.pypa.io/en/latest/index.html).
3. Install requirements by command: `pip install -r requirements.txt`.
5. Modify the parameters in the main functions according to your dataset and preferences.
6. Run the scripts.

## Acknowledgments

This project draws inspiration from the research paper "A novel group recommender system based on members’ influence and leader impact" by Reza Barzegar Nozari and Hamidreza Koohi. Their innovative approach to group recommendations has been implemented here to demonstrate its practical applications.

## Reference
Reza Barzegar Nozari and Hamidreza Koohi. (2020). A novel group recommender system based on members’ influence and leader impact. Knowledge-Based Systems, 205, 106296. https://doi.org/10.1016/j.knosys.2020.106296

## License

This project is licensed under the MIT License.

Feel free to explore, experiment, and adapt the code to your specific use case. If you find this repository helpful, consider citing the paper and giving credit to the authors in your work.
