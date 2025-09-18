import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("IRIS.csv")

# 2. Encode target column (species)
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# 3. Features (X) and Target (y)
X = df.drop("species", axis=1)
y = df["species"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train multiple models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Save Graphs separately

# Pairplot
sns.pairplot(df, hue="species")
plt.savefig("pairplot.png")
plt.close()

# Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.close()

# Accuracy Comparison
plt.bar(accuracies.keys(), accuracies.values(), color=["blue","green","orange"])
plt.title("Model Accuracy Comparison on Iris Dataset")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.savefig("accuracy_comparison.png")
plt.close()
