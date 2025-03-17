# Vector Search at Scale: Pinecone, FAISS, and Elasticsearch Deep Dive  

## Overview  
Vector search is the backbone of **retrieval-augmented generation (RAG)**, AI-powered recommendations, and multimodal search.  
This repository explores **how large-scale AI systems retrieve billions of high-dimensional vectors efficiently**, leveraging:  

✅ **Pinecone** – Fully managed, scalable vector search  
✅ **FAISS** – GPU-optimized high-performance ANN search  
✅ **Elasticsearch** – Hybrid keyword + vector search  

## Topics Covered  
- **HNSW, IVF-PQ, and quantization techniques**  
- **Scaling ANN search to billions of vectors**  
- **Sharding, distributed execution, and query routing**  
- **Performance trade-offs: speed vs. memory vs. accuracy**  
- **Real-world AI applications using vector databases**  

## Code Examples  
This repo includes **fully working examples** for:  
- **Pinecone: Scalable billion-scale vector search with metadata filtering**  
- **FAISS: GPU-accelerated ANN search with IVF-PQ indexing**  
- **Elasticsearch: Hybrid BM25 + dense vector search for enterprise AI**  

## Code Files and Their Descriptions  

### **Pinecone Scripts**  
- **[`pinecone/create_pinecone_index.py`](pinecone/create_pinecone_index.py)** – Creates a Pinecone index optimized for large-scale vector search.  
- **[`pinecone/insert_vectors.py`](pinecone/insert_vectors.py)** – Inserts vectors with metadata for structured filtering.  
- **[`pinecone/query_vectors.py`](pinecone/query_vectors.py)** – Queries the Pinecone index using a vector and metadata filters.  
- **[`pinecone/pinecone_bulk_ingestion.py`](pinecone/pinecone_bulk_ingestion.py)** – High-performance parallel vector ingestion for handling large-scale datasets.  
- **[`pinecone/pinecone_multi_index_routing.py`](pinecone/pinecone_multi_index_routing.py)** – Dynamically routes queries to the appropriate Pinecone index based on workload type.  
- **[`pinecone/pinecone_hybrid_search.py`](pinecone/pinecone_hybrid_search.py)** – Implements hybrid search combining vector retrieval, metadata filtering, and custom re-ranking.  
- **[`pinecone/pinecone_query_expansion.py`](pinecone/pinecone_query_expansion.py)** – Expands queries dynamically to improve recall and retrieval quality.  
- **[`pinecone/pinecone_query_preprocessing.py`](pinecone/pinecone_query_preprocessing.py)** – Preprocesses query vectors using dimensionality reduction (PCA) and normalization for optimized search.  

### **FAISS Scripts**  
- **[`faiss/setup_faiss_ivfpq.py`](faiss/setup_faiss_ivfpq.py)** – Builds an IVF-PQ FAISS index for efficient billion-scale vector search.  
- **[`faiss/query_faiss_ivfpq.py`](faiss/query_faiss_ivfpq.py)** – Loads and queries the FAISS index for approximate nearest neighbor (ANN) search.  
- **[`faiss/faiss_gpu_search.py`](faiss/faiss_gpu_search.py)** – Performs high-speed vector search using GPU-accelerated FAISS indexing.  

### **Elasticsearch Scripts**  
- **[`elasticsearch/setup_elasticsearch.py`](elasticsearch/setup_elasticsearch.py)** – Creates an Elasticsearch index supporting hybrid text + vector search.  
- **[`elasticsearch/insert_elasticsearch.py`](elasticsearch/insert_elasticsearch.py)** – Inserts text-based documents with vector embeddings into Elasticsearch.  
- **[`elasticsearch/query_elasticsearch.py`](elasticsearch/query_elasticsearch.py)** – Executes hybrid search queries combining BM25 full-text search with vector similarity matching.  

### **Miscellaneous**  
- **[`pinecone_agent_vipas.py`](pinecone_agent_vipas.py)** – Pinecone-related script (specific functionality to be detailed).  
- **[`requirements.txt`](requirements.txt)** – Contains dependencies required to run all Pinecone, FAISS, and Elasticsearch examples.  

---

## 🔗 Full Technical Deep Dive  
For a **detailed breakdown of architectures, performance optimizations, and best practices**, check out the full article:  
👉 **[Infrastructure No One Talks About: How Vector Search Makes Gen AI Work](https://www.linkedin.com/pulse/infrastructure-one-talks-how-vector-search-makes-gen-ai-vinay-jayanna-ka2bc)**  

## Getting Started  
Clone the repository and run the examples in Python:  
```bash
git clone https://github.com/your-repo/vector-search-gen-ai
cd vector-search-gen-ai
pip install -r requirements.txt
