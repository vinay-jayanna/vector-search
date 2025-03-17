import faiss
import numpy as np

# Generate 1M random 128-dimensional vectors
d = 128
num_vectors = 1_000_000
vectors = np.random.random((num_vectors, d)).astype('float32')

# Create an IVF-PQ index optimized for billion-scale search
nlist = 100  # Number of partitions
m = 16  # Number of sub-quantizers for PQ
nbits = 8  # Bits per quantizer

quantizer = faiss.IndexFlatL2(d)  # Base index for clustering
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(vectors)  # Train the quantizer
index.add(vectors)  # Add vectors to the index

# Save index to disk
faiss.write_index(index, "ivfpq_index.faiss")

print("FAISS IVF-PQ index created and saved successfully.")
