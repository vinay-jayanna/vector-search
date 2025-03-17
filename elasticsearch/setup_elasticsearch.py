from elasticsearch import Elasticsearch

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9200")

# Define index settings with a dense vector field
index_name = "vector_index"
settings = {
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "vector": {"type": "dense_vector", "dims": 768}
        }
    }
}

# Create the index
es.indices.create(index=index_name, body=settings)
print(f"Elasticsearch index '{index_name}' created successfully.")
