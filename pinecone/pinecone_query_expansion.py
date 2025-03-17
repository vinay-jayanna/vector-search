import pinecone

# Initialize Pinecone client
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("query-expansion-index")

# Query expansion function based on memory-aware optimization
def expand_query(original_vector):
    expansion_terms = [
        [v * 1.05 for v in original_vector],  # Slightly increase weight
        [v * 0.95 for v in original_vector]   # Slightly decrease weight
    ]
    return [original_vector] + expansion_terms  # Include original

# Query processing
query_vector = [0.15, 0.22, 0.33, ...]
expanded_queries = expand_query(query_vector)

# Aggregate results from multiple query variants
all_results = []
for vector in expanded_queries:
    results = index.query(vector, top_k=5)
    all_results.extend(results["matches"])

# Merge and return top results
top_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:5]
print("Final Expanded Query Results:", top_results)
