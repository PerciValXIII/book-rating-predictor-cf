# DATA 606 - Project 2: Collaborative Filtering for Book Rating Prediction

**Author:** Swattik Maiti  
**UID:** 121079367

---

## Overview

This project implements a **User-User Collaborative Filtering** algorithm to predict user ratings for books using the Book-Crossings dataset. The goal is to estimate how a user might rate a book they haven't read, based on their historical preferences and the preferences of similar users. Model performance is evaluated using the **Mean Absolute Difference (MAD)** metric on a held-out test set.

---

## Algorithm Description & Implementation Choices

The collaborative filtering strategy used is **User-User Similarity-based Recommendation**. The key idea is to find users similar to a given user (based on rating behavior), and use their ratings to predict unknown ratings for that user.

### Key Implementation Steps & Choices:

1. **Data Loading**
   - Loaded `Users.csv`, `Books.csv`, and `Ratings.csv` using `pandas`.
   - Set `User-ID` and `ISBN` as indices where appropriate for efficient lookups.

2. **Data Preprocessing**
   - Removed implicit ratings (`Book-Rating == 0`) as they don’t reflect user preference.
   - Filtered out users and books with fewer than **5 ratings** to reduce noise and sparsity.

3. **Encoding**
   - Used `LabelEncoder` to convert `User-ID` and `ISBN` into zero-based integer indices.
   - Encoders were **fit only on the training data** to avoid data leakage, and test values were filtered to match.

4. **Train/Test Split**
   - Randomly split `(User-ID, ISBN)` pairs into **75% training / 25% testing** using `train_test_split` with a fixed `random_state` for reproducibility.
   - Ensured all test users and books exist in the training set to avoid cold-start issues.

5. **User-Item Matrix**
   - Constructed a **sparse user-item matrix** (`csr_matrix`) with shape `[n_users x n_books]`, populated with ratings.

6. **Similarity Calculation**
   - Computed **cosine similarity** between user vectors using `sklearn.metrics.pairwise.cosine_similarity`.
   - Replaced any `NaN` or infinite similarity values with `0` for numerical stability.

7. **Prediction Function**
   - Implemented `predict_rating_user_user()`:
     - For a given `(user, book)` pair, identify users who have rated the book.
     - Select the top-`k` most similar users using precomputed similarity scores.
     - Compute a **weighted average** of the neighbors’ ratings, weighted by their similarity scores.
     - Applied fallbacks using the global mean rating for cases with no available neighbors or zero similarity sum.
     - Clipped predictions to the valid range `[1, 10]`.

8. **Evaluation**
   - Used **Mean Absolute Difference (MAD)** between predicted and actual ratings on the test set.
   - Implemented evaluation over the full test set for each configuration.

9. **Hyperparameter Analysis**
   - Evaluated performance for different `k` values: **5, 10, 15, 20, 50, 100**.
   - Also varied training set size from **60% to 90%** in 5% increments to study how performance scales with data.

---

## Summary of Results

- **Optimal neighborhood size** was found to be **k = 20**, achieving the lowest MAD (~1.5312 on the 75/25 split).
- **Increasing the training ratio** consistently improved performance.
- The best performance was observed at **90% training / 10% testing** with MAD ~**1.5140**.
- Results indicate that collaborative filtering benefits from both:
  - a sufficiently **large neighborhood** (top-k users),
  - and **more historical rating data**.

---

## Instructions to Run Code

1. Ensure the following data files are available in the `data/` directory:
   - `Users.csv`
   - `Books.csv`
   - `Ratings.csv`

2. Install dependencies (via pip):
   ```bash
   pip install -r requirements.txt

3. Run the jupyter notebook --> notebooks/DATA606_P2_Swattik_121079367.ipynb

---

## Dependencies

The following Python libraries are required:

* `pandas`
* `os` 
* `numpy`
* `scikit-learn` 
* `scipy` 
* `matplotlib`
* `seaborn` 
* `tqdm` 

