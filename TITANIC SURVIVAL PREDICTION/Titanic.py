# Importing necessary librarie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# Step 1: Load dataset
# -------------------------
data = pd.read_csv("train.csv")   # make sure train.csv is in the same folder

print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())

# -------------------------
# Step 2: Handle Missing Values
# -------------------------
# Fill missing Age with mean
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Fill Embarked with most frequent value
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop columns not useful for prediction
data.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

# -------------------------
# Step 3: Encode Categorical Data
# -------------------------
# Convert Sex column (male=0, female=1)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# One hot encoding for Embarked
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

print("\nCleaned Data Sample:\n", data.head())

# -------------------------
# Step 4: Split Features & Target
# -------------------------
X = data.drop('Survived', axis=1)   # Features
y = data['Survived']                # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

# -------------------------
# Step 5: Train Models
# -------------------------
# Logistic Regression
logistic_model = LogisticRegression(max_iter=300, solver='liblinear')
logistic_model.fit(X_train, y_train)
log_pred = logistic_model.predict(X_test)

# K-Nearest Neighbors (new model added)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# -------------------------
# Step 6: Evaluate Models
# -------------------------
print("\nModel Results:")

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

print("\nConfusion Matrix (KNN):\n", confusion_matrix(y_test, knn_pred))
print("\nClassification Report (KNN):\n", classification_report(y_test, knn_pred))

# -------------------------
# Step 7: Visualization
# -------------------------
# Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title("Survival Distribution by Gender")
plt.savefig("gender_survival.png")
plt.show()

# Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("Survival Distribution by Passenger Class")
plt.savefig("pclass_survival.png")
plt.show()

# Age Distribution
plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']],
         bins=15, stacked=True, label=['Survived','Not Survived'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution: Survived vs Not Survived')
plt.savefig("age_distribution.png")
plt.legend()
plt.show()