import pinecone

# Initialize Pinecone client
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# Connect to the existing index
index = pinecone.Index("billion-scale-index")

# Insert vectors with metadata for filtering
vectors = [
    ("vec1", [0.1, 0.2, 0.3, 0.4], {"category": "AI Research"}),
    ("vec2", [0.5, 0.1, 0.9, 0.2], {"category": "Healthcare"})
]
index.upsert(vectors)

print("Vectors inserted successfully.")
