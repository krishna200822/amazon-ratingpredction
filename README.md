#  Amazon Product Rating Prediction (ML Project)

Overview

This project focuses on predicting product ratings from an Amazon dataset using machine learning. The goal is to understand how pricing, discounts, and product engagement influence customer ratings.

---

Problem Statement

Can we predict a product’s rating based on its pricing, discount, and popularity features?

---

Dataset Description

The dataset contains Amazon product information including:

* Product details (name, category)
* Pricing (`actual_price`, `discounted_price`)
* Discount information (`discount_percentage`)
* User engagement (`rating_count`)
* Target variable: **`rating`**

---

 Data Preprocessing

The dataset contained messy string values and required cleaning:

* Removed currency symbols (₹) and commas from price columns
* Converted percentage values to numeric
* Handled missing values using `dropna()`
* Converted all relevant columns to numeric types

---

 Feature Engineering

New features were created to improve model performance:

* `price_diff` → Difference between actual and discounted price
* `discount_ratio` → Relative discount level
* `price_bucket` → Categorized prices (cheap, mid, expensive)
* `rating_count_log` → Log transformation to reduce skew

---

 Model Used

### 1. Linear Regression

A baseline model to understand linear relationships.

### 2. Random Forest Regressor (optional improvement)

Used to capture non-linear patterns in the data.

---

# Model Evaluation

Metrics used:

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**

Visualization:

* Actual vs Predicted scatter plot
* Error distribution histogram

---

Results & Insights

* Ratings are heavily clustered between **4.0 – 4.3**
* Model predictions are also concentrated → indicates low variance in target
* Pricing and discount alone are **weak predictors of rating**
* Model shows **underfitting**, suggesting missing important features (e.g., brand, quality, reviews)

---

#  Limitations

* Dataset lacks strong predictive features (like brand or sentiment)
* Ratings have low variance → harder to predict accurately
* Text data (reviews) not utilized

---

# Future Improvements

* Use NLP on review text for better prediction
* Add more meaningful features (brand, sentiment score)
* Try advanced models (XGBoost, Gradient Boosting)
* Perform hyperparameter tuning

---

#  Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

# Project Structure

```
amazon-ml-project/
│
├── data/
│   └── amazon.csv
│
├── project.py
├── README.md
└── requirements.txt
```

---

#  Conclusion

This project demonstrates a complete ML pipeline:

* Data cleaning
* Feature engineering
* Model training
* Evaluation

While performance is limited due to dataset constraints, the project highlights the importance of feature quality over model complexity.

---

# Author

Krishna
