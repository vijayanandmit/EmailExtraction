import os
import glob
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define the directory where email text files are stored.
# Update this path to your folder location containing the email files.
email_directory = "./"  # e.g., "./emails/"
file_pattern = os.path.join(email_directory, "*.txt")
file_list = glob.glob(file_pattern)

# Load emails and keep track of filenames.
emails = []
filenames = []
for file_path in file_list:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        emails.append(content)
        filenames.append(os.path.basename(file_path))

# Preprocessing function: Remove common email header lines.
def preprocess_email(text):
    # Remove lines that start with standard header fields.
    lines = text.splitlines()
    filtered_lines = [line for line in lines if not re.match(r"^(Date|From|To|Subject):", line)]
    # Join the remaining lines and remove extra spaces.
    text_clean = " ".join(filtered_lines)
    return text_clean

# Apply preprocessing to each email.
emails_clean = [preprocess_email(email) for email in emails]

# Vectorize the email texts using TF-IDF.
vectorizer = TfidfVectorizer(stop_words=None, max_df=0.8)
X = vectorizer.fit_transform(emails_clean)

# Define number of clusters.
# Here, we assume two clusters: one for automated notifications and one for other email types.
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Create a DataFrame to display file names and their assigned cluster.
df = pd.DataFrame({"Filename": filenames, "Cluster": labels})
print("Cluster Assignments:")
print(df)

# Identify top terms per cluster to interpret the clustering.
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for cluster_num in range(num_clusters):
    print(f"\nCluster {cluster_num} top terms:")
    top_terms = [terms[ind] for ind in order_centroids[cluster_num, :10]]
    print(", ".join(top_terms))

# (Optional) Plot the clusters using PCA for dimensionality reduction.
pca = PCA(n_components=2, random_state=42)
reduced_data = pca.fit_transform(X.toarray())

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis")
plt.title("Email Clusters (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, ticks=range(num_clusters), label="Cluster")
plt.show()

# Future Integration: 
# Once clusters are identified, you can map clusters to automated response templates.
# For example:
# cluster_mapping = {0: "Automated Invoice Notification", 1: "Invoice Inquiry"}
# df["Cluster_Label"] = df["Cluster"].map(cluster_mapping)
# print(df)
