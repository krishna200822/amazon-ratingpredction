import sys
print(sys.executable)
import pandas as pd

# Load dataset
df = pd.read_csv("amazon.csv")

# -----------------------------
# 🔍 Inspect data (important)
# -----------------------------
print(df.head())        # preview first rows
print(df.columns)       # check column names
print(df.info())        # check data types

# -----------------------------
# 💰 Fix price columns
# -----------------------------
# Remove ₹ and commas, then convert to float safely
df['discounted_price'] = pd.to_numeric(
    df['discounted_price'].str.replace('₹', '').str.replace(',', ''),
    errors='coerce'   # invalid values → NaN instead of crash
)

df['actual_price'] = pd.to_numeric(
    df['actual_price'].str.replace('₹', '').str.replace(',', ''),
    errors='coerce'
)

# -----------------------------
# 📊 Fix discount percentage
# -----------------------------
# Remove % sign and convert to float
df['discount_percentage'] = pd.to_numeric(
    df['discount_percentage'].str.replace('%', ''),
    errors='coerce'
)

# -----------------------------
# ⭐ Fix rating
# -----------------------------
# Convert rating to float (handles bad values)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# -----------------------------
# 🔢 Fix rating count
# -----------------------------
# Remove commas and convert to float
df['rating_count'] = pd.to_numeric(
    df['rating_count'].str.replace(',', ''),
    errors='coerce'
)

# -----------------------------
# 🧹 Handle missing values
# -----------------------------
# Check how many null values exist
print(df.isnull().sum())

# Drop rows with missing values (simple approach)
df = df.dropna()

# -----------------------------
# ✅ Final check
# -----------------------------
print(df.dtypes)   # confirm all numeric columns are correct
print(df.shape)    # see how much data remains
# -----------------------------
# 💡 Price difference (actual - discounted)
# -----------------------------
# Shows how much discount in absolute value
df['price_diff'] = df['actual_price'] - df['discounted_price']


# -----------------------------
# 💡 Discount ratio
# -----------------------------
# More useful than percentage sometimes (0 to 1 scale)
df['discount_ratio'] = df['discounted_price'] / df['actual_price']


# -----------------------------
# 💡 Price bucket (cheap / mid / expensive)
# -----------------------------
# Helps model understand price segments
df['price_bucket'] = pd.cut(
    df['actual_price'],
    bins=[0, 500, 2000, 10000],
    labels=['cheap', 'mid', 'expensive']
)


# -----------------------------
# 💡 Log of rating count
# -----------------------------
# Reduces skew (important for ML)
import numpy as np
df['rating_count_log'] = np.log1p(df['rating_count'])
print(df[['price_diff', 'discount_ratio', 'price_bucket', 'rating_count_log']].head())
features = [
    'discounted_price',
    'actual_price',
    'discount_percentage',
    'rating_count_log',
    'price_diff',
    'discount_ratio'
]
target = 'rating'
df = pd.get_dummies(df, columns=['price_bucket'], drop_first=True)
X = df[features + ['price_bucket_mid', 'price_bucket_expensive']]
y = df[target]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression

# create model
model = LinearRegression()

# train
model.fit(X_train, y_train)
# predction
y_pred = model.predict(X_test)
# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
# actual vs predcitedx
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted")
plt.show()
# error
plt.hist(y_test - y_pred, bins=20)
plt.title("Error Distribution")
plt.show()