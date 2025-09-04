# 🚢 Titanic Survival Prediction  

This project is part of my **CodSoft Data Science Internship**.  
The goal is to build a machine learning model that predicts whether a passenger on the Titanic survived or not.  

---

## 📌 Project Overview
The Titanic dataset is a beginner-friendly dataset commonly used in data science.  
It contains information about passengers such as:  
- Age  
- Gender  
- Passenger Class (Pclass)  
- Fare  
- Number of Siblings/Spouses (SibSp)  
- Number of Parents/Children (Parch)  
- Embarked Port (C/Q/S)  
- Survival (Target variable)  

---

## ⚙️ Steps Performed
1. **Data Cleaning**  
   - Handled missing values (Age → mean, Embarked → mode).  
   - Dropped unnecessary columns (Cabin, Ticket, Name).  

2. **Feature Engineering**  
   - Converted categorical data (Sex, Embarked) into numeric.  
   - Prepared features (X) and target (y).  

3. **Model Training**  
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  

4. **Model Evaluation**  
   - Accuracy score  
   - Confusion matrix  
   - Classification report  

5. **Visualization**  
   - Survival by Gender  
   - Survival by Passenger Class  
   - Age Distribution (Survived vs Not Survived)  

---

## 📊 Results
- **Logistic Regression Accuracy**: ~78%  
- **KNN Accuracy**: ~73–76%  
- Female passengers had higher survival rates.  
- 1st Class passengers had higher survival probability compared to 3rd Class.  
- Children had better survival chances than adults.  

---

## 📷 Sample Visualizations
*(Attach your saved graphs here or add them in the repo)*  

- `gender_survival.png`  
- `pclass_survival.png`  
- `age_distribution.png`  

---

## 📂 Repository Structure
