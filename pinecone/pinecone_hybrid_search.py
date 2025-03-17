import pinecone

# Initialize Pinecone client
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("hybrid-vector-index")

# Query vector
query_vector = [0.15, 0.22, 0.33, ...]

# Perform initial vector search with metadata filtering
initial_results = index.query(query_vector, top_k=10, filter={"category": "AI Research"})

# Custom re-ranking logic (e.g., boosting based on additional metadata)
def rerank_results(results):
    reranked = sorted(results["matches"], key=lambda x: x["metadata"].get("relevance_score", 0), reverse=True)
    return reranked[:5]  # Return top 5 after re-ranking

# Apply re-ranking
final_results = rerank_results(initial_results)
print("Final Re-Ranked Results:", final_results)
