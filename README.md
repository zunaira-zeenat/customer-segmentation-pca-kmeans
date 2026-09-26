# Customer Segmentation using PCA & K-Means 📊

A Machine Learning project that performs **customer segmentation using PCA (Principal Component Analysis) and K-Means Clustering**.

The project uses customer behavior data, applies feature scaling and PCA for dimensionality reduction, and then uses K-Means to identify groups of similar customers.

## 🎯 Project Objective

The main objective of this project is to:

* Analyze customer behavior
* Reduce multiple features using PCA
* Identify similar groups of customers
* Perform customer segmentation using K-Means
* Visualize the resulting customer clusters

## 🧠 Machine Learning Techniques

### PCA — Principal Component Analysis

PCA is used to reduce the dimensionality of the customer dataset.

The original dataset contains multiple customer-related features. PCA transforms these features into two principal components:

* **PC1**
* **PC2**

This makes the data easier to visualize while retaining important patterns.

### K-Means Clustering

K-Means is an **unsupervised Machine Learning algorithm** that groups similar data points into clusters.

In this project, K-Means is used to divide customers into **3 clusters** based on their behavior.

## 📊 Dataset

The dataset contains 30 customer records with the following features:

| Feature                  | Description                             |
| ------------------------ | --------------------------------------- |
| Customer_ID              | Unique customer identifier              |
| Age                      | Customer age                            |
| Annual_Income            | Annual income                           |
| Monthly_Spending         | Monthly spending                        |
| Purchases_Per_Month      | Number of monthly purchases             |
| Website_Visits_Per_Month | Monthly website visits                  |
| Discount_Usage_Percent   | Percentage of purchases using discounts |

`Customer_ID` is used only as an identifier and is not included in the PCA or K-Means features.

## 🔄 Project Workflow

```text
Customer Data
     ↓
Data Exploration
     ↓
Feature Selection
     ↓
Feature Scaling
     ↓
PCA
     ↓
2 Principal Components
     ↓
K-Means Clustering
     ↓
Customer Segments
     ↓
Visualization
```

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Google Colab

## 📌 Project Steps

### 1. Data Exploration

The dataset is explored using Pandas functions such as:

```python
df.head()
df.shape
df.columns
df.info()
df.describe()
```

### 2. Feature Selection

Customer behavior features are selected for Machine Learning:

```python
X = df[
    [
        "Age",
        "Annual_Income",
        "Monthly_Spending",
        "Purchases_Per_Month",
        "Website_Visits_Per_Month",
        "Discount_Usage_Percent"
    ]
]
```

### 3. Feature Scaling

`StandardScaler` is used to standardize the selected features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 4. PCA

PCA reduces the six selected features to two principal components.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### 5. K-Means Clustering

K-Means is applied to the PCA-transformed data.

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)

model.fit(X_pca)

pca_df["Cluster"] = model.labels_
```

### 6. Customer Segmentation Visualization

The clusters are visualized using PC1 and PC2.

```python
import matplotlib.pyplot as plt

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=pca_df["Cluster"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Segmentation using PCA and K-Means")
plt.show()
```

## 📈 Results

The K-Means algorithm assigns each customer to one of **three clusters**.

PCA allows these customer groups to be visualized in a two-dimensional space using PC1 and PC2.

The resulting clusters can be analyzed to understand differences in customer behavior, such as spending, purchasing frequency, website engagement, and discount usage.

## 💡 Key Learning

This project helped me practice:

* Data exploration
* Feature selection
* Feature scaling
* PCA
* Explained variance
* K-Means clustering
* Customer segmentation
* Data visualization
* Unsupervised Machine Learning

## 🚀 Future Improvements

* Use the **Elbow Method** to determine an appropriate number of clusters
* Analyze the characteristics of each customer cluster
* Use a larger real-world customer dataset
* Add interactive visualizations
* Build a simple dashboard for customer segmentation
* Compare clustering results with and without PCA

## 📁 Repository Structure

```text
customer-segmentation-pca-kmeans/
│
├── customer_segmentation_pca_kmeans.ipynb
├── customer_behavior_dataset.csv
└── README.md
```

## 👩‍💻 Author

**Zunaira Zeenat**

BS Information Technology Student
Interested in Machine Learning, AI, and Data Science.
