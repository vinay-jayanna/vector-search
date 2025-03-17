from elasticsearch import Elasticsearch

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9200")

# Define a hybrid query with BM25 and KNN search
query_body = {
    "query": {
        "bool": {
            "should": [
                {"match": {"text": "AI research document"}},
                {"knn": {"vector": {"vector": [0.1, 0.4, 0.3, 0.2], "k": 10}}}
            ]
        }
    }
}

# Execute search query
response = es.search(index="vector_index", body=query_body)

# Display results
for hit in response["hits"]["hits"]:
    print(f"Document ID: {hit['_id']}, Score: {hit['_score']}, Text: {hit['_source']['text']}")
