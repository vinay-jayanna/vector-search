import pinecone

# Initialize Pinecone client
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# Create a scalable index optimized for billion-scale vector search
pinecone.create_index(
    name="billion-scale-index",
    dimension=1536,  # Vector dimension
    metric="cosine",  # Similarity metric
    pods=50,  # Number of pods to handle dataset scale
    pod_type="p2.x1",  # High-memory pod type for optimal performance
    index_type="ivf_pq"  # Use IVF-PQ for large-scale data compression
)

print("Pinecone index 'billion-scale-index' created successfully.")
