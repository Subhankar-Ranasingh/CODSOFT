import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import zipfile
import os

# ===============================
# Step 1: Load Dataset
# ===============================
df = pd.read_csv("IMDb Movies india.csv", encoding="latin1")

# Drop rows with missing essential values
df = df.dropna(subset=['Year','Duration','Votes','Rating','Genre','Director','Actor 1','Actor 2','Actor 3'])

# ===============================
# Step 2: Clean Column Names
# ===============================
df.columns = df.columns.str.strip().str.replace(" ", "").str.lower()

# ===============================
# Step 3: Clean Numeric Columns
# ===============================
df['year'] = pd.to_numeric(df['year'].astype(str).str.replace(r'[()]','', regex=True), errors='coerce')
df['duration'] = pd.to_numeric(df['duration'].astype(str).str.replace(r'[^\d]','', regex=True), errors='coerce')
df['votes'] = pd.to_numeric(df['votes'].astype(str).str.replace(',',''), errors='coerce')

df = df.dropna()

# ===============================
# Step 4: Clean & Encode Categorical Columns
# ===============================
categorical_cols = ['genre','director','actor1','actor2','actor3']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()                    # Remove spaces
    df[col] = df[col].str.replace(r'[^a-zA-Z0-9 ]','', regex=True)  # Remove special chars like #
    df[col] = LabelEncoder().fit_transform(df[col])

# ===============================
# Step 5: Features and Target
# ===============================
X = df[['year','duration','genre','votes','director','actor1','actor2','actor3']]
y = df['rating']

# Ensure numeric columns for correlation
numeric_cols = X.columns.tolist() + ['rating']

# ===============================
# Step 6: Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# Step 7: Train Models
# ===============================
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ===============================
# Step 8: Evaluation
# ===============================
results_text = []
results_text.append("📊 Linear Regression Results:")
results_text.append(f"MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
results_text.append(f"R²: {r2_score(y_test, y_pred_lr):.4f}\n")

results_text.append("🌳 Random Forest Results:")
results_text.append(f"MSE: {mean_squared_error(y_test, y_pred_rf):.4f}")
results_text.append(f"R²: {r2_score(y_test, y_pred_rf):.4f}\n")

with open("results.txt","w",encoding="utf-8") as f:
    f.write("\n".join(results_text))

# ===============================
# Step 9: Visualizations
# ===============================
# 1️⃣ Actual vs Predicted Ratings
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color="blue")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("🎬 Actual vs Predicted Ratings (Random Forest)")
plt.savefig("plot_actual_vs_predicted.png")
plt.close()

# 2️⃣ Genre-wise Average Rating
genre_avg = df.groupby('genre')['rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,6))
genre_avg.plot(kind='bar', color='green')
plt.ylabel("Average Rating")
plt.title("🎞️ Genre-wise Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_genre_avg.png")
plt.close()

# 3️⃣ Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("📊 Correlation Heatmap")
plt.tight_layout()
plt.savefig("plot_correlation.png")
plt.close()

# ===============================
# Step 10: Save all outputs to ZIP
# ===============================
with zipfile.ZipFile("output.zip","w") as zipf:
    zipf.write("results.txt")
    zipf.write("plot_actual_vs_predicted.png")
    zipf.write("plot_genre_avg.png")
    zipf.write("plot_correlation.png")

# Cleanup individual files
os.remove("results.txt")
os.remove("plot_actual_vs_predicted.png")
os.remove("plot_genre_avg.png")
os.remove("plot_correlation.png")

print("✅ All results and plots saved in output.zip")
