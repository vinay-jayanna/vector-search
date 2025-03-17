import faiss
import numpy as np

# Load the trained FAISS index
index = faiss.read_index("ivfpq_index.faiss")

# Generate a random query vector
d = 128
query_vector = np.random.random((1, d)).astype('float32')

# Perform a nearest neighbor search
D, I = index.search(query_vector, k=5)
print("Top-5 Nearest Neighbors:", I)
