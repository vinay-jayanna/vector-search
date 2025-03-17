import pinecone
import numpy as np
import concurrent.futures

# Initialize Pinecone client
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# Connect to the existing index
index = pinecone.Index("billion-scale-index")

# Generate a batch of high-dimensional random vectors
num_vectors = 100000  # Large-scale ingestion
dim = 1536  # Vector dimension
batch_size = 1000  # Batch processing

# Function to insert vectors in parallel
def upsert_vectors(start_idx):
    vectors = [(f"vec_{i}", np.random.random(dim).tolist(), {"category": "AI Research"}) 
               for i in range(start_idx, start_idx + batch_size)]
    index.upsert(vectors)

# Use ThreadPoolExecutor for parallel ingestion
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(upsert_vectors, i) for i in range(0, num_vectors, batch_size)]
    concurrent.futures.wait(futures)

print(f"Successfully ingested {num_vectors} vectors asynchronously.")
