from elasticsearch import Elasticsearch
import numpy as np

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9200")

# Define document with text and vector embedding
doc_1 = {
    "text": "This is an AI research document",
    "vector": list(np.random.random(768))  # Random 768-dimensional vector
}

doc_2 = {
    "text": "Healthcare research on deep learning",
    "vector": list(np.random.random(768))  # Random 768-dimensional vector
}

# Insert documents
es.index(index="vector_index", id=1, body=doc_1)
es.index(index="vector_index", id=2, body=doc_2)

print("Documents inserted successfully.")
