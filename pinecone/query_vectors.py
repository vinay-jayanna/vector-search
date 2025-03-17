import pinecone

# Initialize Pinecone client
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# Connect to the existing index
index = pinecone.Index("billion-scale-index")

# Querying the index with metadata filtering
query_vector = [0.15, 0.22, 0.33, 0.44]
response = index.query(query_vector, top_k=5, filter={"category": "AI Research"})

# Display results
for match in response["matches"]:
    print(f"Match ID: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}")
