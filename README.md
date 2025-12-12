# Actor Recommendation System using Collaborative Filtering (SVD)

This project implements an **Actor Recommendation System** that predicts which actors a user is most likely to prefer based on their historical movie ratings.  
The system uses **Collaborative Filtering**, specifically **Singular Value Decomposition (SVD)** from the Surprise library, to learn user–actor interaction patterns and generate personalized Top-N recommendations.

---

## Features

- Builds a **user–actor rating matrix** by merging MovieLens ratings with actor metadata extracted from TMDb.
- Uses **SVD matrix factorization** to model latent preferences.
- Supports **Top-N actor recommendations** for any user.
- Evaluated using:
  - **Precision@k**
  - **Recall@k**
  - **NDCG@k**
- Clean workflow from data extraction → preprocessing → model training → evaluation.

---

## Dataset Used

### **1. MovieLens Ratings Dataset**
- Contains: `userId`, `movieId`, `rating`, `timestamp`
- User’s rating history forms the base interactions.

### **2. TMDb Actor Metadata**
Collected via API:
- For each movie → list of top 5 actors  
- Stored as `movie_actors.csv`

### **3. Movie–Actor Relationship Dataset**
Created via preprocessing:
- `movie_id_db`, `movie_title`, `actor_name`
- Each row = **one actor per movie**  
- Used to map user ratings to actors.

### **4. User–Actor Ratings**
After merging:
- `userId`
- `actor_name`
- `average_rating`  
(Mean rating a user gives to movies an actor appears in.)

---

## Workflow / Pipeline
MovieLens Ratings + TMDb Actors
│
▼
Create movie_actor_relationships.csv
│
▼
Merge → Build user_actor_ratings.csv (user → actor avg rating)
│
▼
Train SVD Model using Surprise Library
│
▼
Predict Ratings → Generate Top-N Actor Recommendations
│
▼
Evaluate using Precision@k, Recall@k, NDCG@k


---

## Model Used — SVD (Singular Value Decomposition)

SVD factorizes the interaction matrix into:

- **User latent factors**
- **Actor latent factors**
- **Bias terms** (user bias + actor bias)


The model is trained via **Stochastic Gradient Descent (SGD)** with regularization to prevent overfitting.

---

## Evaluation Metrics

### **Precision@10**
“How many of the top 10 recommended actors were relevant?”

###  **Recall@10**
“How many relevant actors were captured in the top 10 recommendations?”

### **NDCG@10**
“Were relevant actors ranked at the top of the recommendation list?”

### **Sample Output (User 5):**
Precision@10 = 0.90
Recall@10 = 0.32
NDCG@10 = 0.93

### **Conclusion**
- Very **accurate recommendations** (high precision)
- Limited **coverage** (moderate recall)
- **Excellent ranking quality** (high NDCG)

---


---

## Installation

```bash
pip install pandas numpy scikit-learn surprise

License
This project is for educational and academic purposes.

Author
Shailja Patil
Master’s Student — CSE, IIT Kharagpur
Special focus: ML, Data Analytics, and Recommender Systems


