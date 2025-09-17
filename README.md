# Mad-Data Project: Churn Prediction Analysis

## 📂 Project Overview

This project focuses on **predicting customer churn** using data from multiple sources. The analysis combines **data exploration, preprocessing, clustering, machine learning modeling, and feature importance interpretation** to provide actionable insights for Atlassian’s customer retention strategy.

### Project Structure

```
Mad-Data/
│
├── Dataset/
│   ├── billing.csv
│   ├── events.csv
│   ├── sessions.csv
│   ├── users.csv
│
├── EDA.pbix                      # Power BI analysis file
├── MAD DATA - MOI B.pdf          # Presentation slides / final submission
├── merged_data.csv               # Combined dataset used in analysis
├── Modelling.ipynb               # Jupyter notebook with full code
└── README.md                     # Project documentation
```

---

## 🧰 Tools & Libraries

This project uses **Python** and the following libraries:

* Data manipulation: `pandas`, `numpy`
* Visualization: `matplotlib`, `seaborn`
* Machine Learning: `scikit-learn`, `xgboost`
* Imbalanced learning: `imblearn` (SMOTEENN)
* Model interpretation: `shap`
* Clustering: `KMeans`, `PCA`

---

## 📝 Project Steps

### 1. Data Loading & Preparation

* Load four datasets: `billing.csv`, `events.csv`, `sessions.csv`, and `users.csv`.
* Aggregate and summarize each dataset by `user_id`:

  * **Sessions:** total sessions, device & OS diversity, average session length
  * **Events:** total events, distinct features/actions, average duration, success rate
  * **Billing:** average MRR, total seats, discounts, overdue invoices, most common plan
* Merge all summaries into `merged_data.csv`.
* Handle missing values and drop irrelevant columns.

### 2. Exploratory Data Analysis (EDA)

* Analyze user distribution by plan type and enterprise status.
* Identify numeric and categorical feature distributions.
* Check correlations among numeric features and remove highly correlated variables to avoid redundancy.

### 3. Clustering

* Perform **K-Means clustering** on standardized numeric features.
* Determine optimal number of clusters using **Elbow Method** (k=3 chosen).
* Visualize clusters with **PCA 2D projection**.
* Analyze patterns in each cluster:

  * Numeric summary statistics
  * Categorical feature distributions
  * Churn distribution per cluster

### 4. Machine Learning Modeling

#### Logistic Regression (per cluster)

* Preprocess numeric and categorical features using **ColumnTransformer**.
* Handle imbalanced data with **SMOTEENN**.
* Perform **cross-validated predictions**.
* Compute **F1 score**, **confusion matrix**, and **classification report**.
* Analyze **feature importance** via coefficients.

#### XGBoost Classifier (per cluster)

* Train **XGBoost** using cross-validation.
* Evaluate metrics for each cluster.
* Extract **SHAP values** to identify top features influencing churn predictions.

### 5. Cluster Feature Patterns

* Calculate mean values for key features per cluster:

  * `distinct_features`, `distinct_action`, `total_events`, `os_diversity`, `avg_mrr`
* This helps identify customer segments and churn risk patterns.

---

## 📊 Visualizations

* **Bar plots**: User distribution by plan type and enterprise status.
* **Boxplots & histograms**: Numeric feature distributions by cluster.
* **Countplots**: Categorical feature distributions by cluster.
* **PCA scatter plots**: Visualize clusters in 2D space.
* **SHAP summary plots**: Identify top 5 features impacting churn predictions.

---

## 🧩 How to Run

1. Clone the repository:

```bash
git clone <your-repo-url>
cd Mad-Data
```

2. Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

3. Open and run the notebook:

```bash
jupyter notebook Modelling.ipynb
```

* Follow the cells step-by-step to reproduce **data preparation, clustering, modeling, and visualizations**.

---

## 🏆 Insights

* Cluster analysis helps identify **high-risk churn customers**.
* Logistic Regression and XGBoost models achieve high accuracy and precision, especially focusing on churned users.
* SHAP values highlight the **most influential features** for churn prediction:

  * E.g., total events, number of distinct features/actions, OS diversity, and average MRR.

---

## 📄 Deliverables

* `merged_data.csv` – combined dataset for analysis.
* `Modelling.ipynb` – full Python code for preprocessing, clustering, modeling, and visualization.
* `EDA.pbix` – Power BI dashboard for additional analysis.
* `MAD DATA - MOI BOT.pdf` – final presentation slides.

---

## 👤 Author

**Ahmad Nasiruddin Dzulkifli**
Email: nasirdzul@gmail.com

**Muhammad Ismail Putra Zaidi**
Email: muhdismailputra6@gmail.com

**Muhammad Izzul Hafizi Roslan**
Email: hafizir2003@gmail.com

**Imran Fareez Azmy**
Email: iazm0290@uni.sydney.edu.au