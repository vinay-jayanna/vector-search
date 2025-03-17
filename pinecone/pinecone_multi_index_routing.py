import pinecone

# Initialize Pinecone client
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# List of multiple indexes based on workload
index_map = {
    "text-search": "text_vector_index",
    "image-search": "image_vector_index",
    "multimodal-search": "hybrid_vector_index"
}

# Function to query the correct index dynamically
def query_index(search_type, query_vector, top_k=5):
    index_name = index_map.get(search_type)
    if not index_name:
        raise ValueError(f"Invalid search type: {search_type}")
    
    index = pinecone.Index(index_name)
    results = index.query(query_vector, top_k=top_k)
    return results

# Example usage
query_vector = [0.12, 0.45, 0.33, ...]  # Sample vector
search_results = query_index("text-search", query_vector)
print("Query Results:", search_results)
