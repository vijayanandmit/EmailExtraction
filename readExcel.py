import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the Excel file into a DataFrame."""
    df = pd.read_excel(file_path)
    return df

def preprocess_emails(df):
    """
    Combine the 'Subject' and 'Body' fields to create a single text column.
    Convert text to lowercase for consistency.
    """
    df['text'] = (df['Subject'].fillna('') + " " + df['Body'].fillna('')).str.lower()
    return df

def vectorize_text(text_data):
    """
    Use TF-IDF vectorization to convert text data into numerical features.
    English stop words are removed.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer

def perform_clustering(X, num_clusters=5):
    """
    Apply KMeans clustering on the TF-IDF features.
    The number of clusters can be adjusted by changing the 'num_clusters' parameter.
    """
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(X)
    clusters = km.labels_
    return clusters, km

def visualize_clusters(X, clusters):
    """
    Use PCA to reduce the feature space to 2 dimensions and visualize the clusters.
    """
    pca = PCA(n_components=2, random_state=42)
    X_reduced = pca.fit_transform(X.toarray())
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title('Email Clusters Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

def main():
    # Set file path to your Excel file
    file_path = 'Emails Extract in Excel_March 13th.xlsx'
    
    # Load and preprocess the data
    df = load_data(file_path)
    print(f"Loaded {len(df)} records.")
    df = preprocess_emails(df)
    
    # Convert email text to numerical features
    X, vectorizer = vectorize_text(df['text'])
    
    # Set the number of clusters (adjust as needed)
    num_clusters = 5
    clusters, km = perform_clustering(X, num_clusters)
    
    # Add cluster labels to the DataFrame
    df['cluster'] = clusters
    print("Cluster distribution:")
    print(df['cluster'].value_counts())
    
    # Visualize clusters using PCA
    visualize_clusters(X, clusters)
    
    # Optionally, save the clustered data to a new Excel file
    output_file = 'clustered_emails.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Clustered emails saved to '{output_file}'.")

if __name__ == "__main__":
    main()
