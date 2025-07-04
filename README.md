# Customer Churn Prediction Classifier

This project aims to predict **customer churn** using supervised machine learning models. It walks through the **complete data science workflow** from data exploration to deployment. The final solution is served through a **Streamlit web app** for interactive usage.

---

## Project Overview

Customer churn — when a customer stops using a product or service — is a critical metric for many businesses. By predicting churn, companies can take proactive measures to improve customer retention and reduce revenue loss.
This project builds a churn prediction model using structured telecom customer data. It compares several machine learning algorithms and deploys the final solution via a lightweight web app.
---

## Technologies Used

- **Python 3**
- **Pandas** – Data manipulation  
- **NumPy** – Numerical computations  
- **Scikit-learn** – Model training & evaluation  
- **Matplotlib & Seaborn** – Visualization  
- **Streamlit** – Web app deployment  
- **Jupyter Notebook** – Experimentation & documentation  
---

## Models Used

The following classification models were trained and evaluated:

- Logistic Regression  
- Support Vector Classifier (SVC)  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  

**Model selection** was based on **accuracy**, with additional consideration of interpretability and performance consistency.
---

## Dataset

- **Source:** [Kaggle - Telecom Customer Churn Dataset](https://www.kaggle.com/datasets/abdullah0a/telecom-customer-churn-insights-for-analysis)  
- **Key Features:**

- CustomerID: Unique identifier for each customer.
- Age: Age of the customer, reflecting their demographic profile.
- Gender: Gender of the customer (Male or Female).
- Tenure: Duration (in months) the customer has been with the service provider.
- MonthlyCharges: The monthly fee charged to the customer.
- ContractType: Type of contract the customer is on (Month-to-Month, One-Year, Two-Year).
- InternetService: Type of internet service subscribed to (DSL, Fiber Optic, None).
- TechSupport: Whether the customer has tech support (Yes or No).
- TotalCharges: Total amount charged to the customer (calculated as MonthlyCharges * Tenure).
- Churn: Target variable indicating whether the customer has churned (Yes or No).

- **Target Variable:** `Churn` (Yes/No)
---

## Project Workflow

1. **Data Extraction:** Downloaded from Kaggle repository  
2. **Data Understanding:** Exploratory Data Analysis (EDA), data types, patterns  
3. **Preprocessing:** Handled nulls, duplicates, encoding categorical features  
4. **Feature Selection:** Identified most relevant features  
5. **Visualization:** Uncovered churn patterns using bar plots, heatmaps, etc.  
6. **Train-Test Split:** Stratified sampling for balanced class distribution  
7. **Scaling:** Used `StandardScaler` for normalization  
8. **Modeling:** Trained and compared multiple classifiers  
9. **Hyperparameter Tuning:** Applied **GridSearchCV** for optimized model performance  
10. **Evaluation:** Compared models using accuracy and confusion matrix  
11. **Deployment:** Built a **Streamlit** app for real-time churn prediction  
---

### Hyperparameter Tuning

To improve model performance and prevent overfitting, **GridSearchCV** was used to tune hyperparameters for models such as:

- **Random Forest** – Tuned `n_estimators`, `max_depth`, `min_samples_split`, and `criterion`
- **SVC** – Tuned `C`, `kernel`, and `gamma`
- **KNN** – Tuned `n_neighbors` and distance metrics
- Cross-validation ensured robust selection of the best parameter combinations based on validation accuracy.
---

## Result

The **Support Vector Classifier** performed the best with high accuracy and generalization with an accuracy score of 88 %. The final Streamlit app allows users to input customer details and receive an instant churn prediction.
---

## Future Enhancements

- Integrate SHAP for model explainability  
- Add precision, recall, and F1-score in evaluation  
- Include cost-sensitive learning or churn probability scores  
- Expand deployment with Docker or cloud hosting (e.g., AWS/GCP)
---

## UI Overview
![image](https://github.com/user-attachments/assets/92e6067f-c78c-40b2-b679-a041f7b1cd21)




