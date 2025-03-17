import faiss
import numpy as np

# Generate 1M random 128-dimensional vectors
d = 128
num_vectors = 1_000_000
vectors = np.random.random((num_vectors, d)).astype('float32')

# Move index to GPU
gpu_resources = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatL2(d)
index_gpu = faiss.index_cpu_to_gpu(gpu_resources, 0, index_cpu)

# Add vectors and perform search
index_gpu.add(vectors)
query_vector = np.random.random((1, d)).astype('float32')
D, I = index_gpu.search(query_vector, k=5)
print("Top-5 Nearest Neighbors (GPU Accelerated):", I)
