import os
import time
import uuid
import numpy as np
import jsonlines
import sentence_transformers
import chromadb

# Constants
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHROMA_PERSIST_DIRECTORY = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/chroma_db/scirex-v2"
CHROMA_COLLECTION_NAME = "all"
FILE_PATH = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/scirex/Preprocessed_Scirex.jsonl"

def sentence_embedding(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts)

def create_chroma_db_local(persist_directory, collection_name):
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection

def insert_to_db(texts, model_name, device, db):
    model = sentence_transformers.SentenceTransformer(model_name, device=device)
    batch_embeddings = []
    batch_texts = []
    start_time = time.time()
    print(f"Total texts to process: {len(texts)}, Device: {device}")
    for i, text in enumerate(texts):
        embedding = sentence_embedding(model, text)[0].tolist()  # Ensures correct shape
        batch_embeddings.append(embedding)
        batch_texts.append(text)

        if i % 100 == 0 or i == len(texts) - 1:
            batch_ids = [str(uuid.uuid1()) for _ in batch_texts]
            db.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                ids=batch_ids
            )
            batch_embeddings = []
            batch_texts = []
            print(f"Inserted {i+1} texts. Time: {time.time() - start_time:.2f}s")
    print(f"Insertion complete. Total time: {time.time() - start_time:.2f}s")

def load_model_offline(model_name, cache_dir=None):
    """Load model from local cache directory for offline use"""
    if cache_dir is None:
        cache_dir = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/chroma_db/agenda"
    
    # First try to load from cache
    local_model_path = os.path.join(cache_dir, model_name.replace("/", "_"))
    
    if os.path.exists(local_model_path):
        print(f"Loading model from cache: {local_model_path}")
        return sentence_transformers.SentenceTransformer(local_model_path, device="cpu")
    else:
        print(f"Model not found in cache. Downloading to: {local_model_path}")
        # Download and save to cache
        os.makedirs(cache_dir, exist_ok=True)
        model = sentence_transformers.SentenceTransformer(model_name, device="cpu", cache_folder=cache_dir)
        model.save(local_model_path)
        return model



def query_llm(devices, query):
    db = create_chroma_db_local(CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME)
    input_texts = []

    with open(FILE_PATH, 'r') as f:
        for item in jsonlines.Reader(f):
            input_texts.append(item["content"])
    print("Total Number of Papers:", len(input_texts))

    # Check if DB is empty
    if len(db.get()["ids"]) == 0:
        print("Database empty. Inserting data...")
        insert_to_db(input_texts, model_name=EMBED_MODEL_NAME, device="cpu", db=db)
    else:
        print("Database already populated. Skipping insert.")

    # Encode the query and retrieve
    model = load_model_offline(EMBED_MODEL_NAME, "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/chroma_db/agenda")
    query_embedding = sentence_embedding(model, [query])  # Returns [[...]]
    results = db.query(query_embeddings=query_embedding, n_results=3)

    if not results["documents"] or not results["documents"][0]:
        return "No results found."

    retrieval_content = '\n'.join(results["documents"][0])
    return retrieval_content

def main(devices, query):
    result = query_llm(devices, query)
    print("\n=== Retrieved ===")
    print(result)

if __name__ == '__main__':
    query = "What is the corresponding EM score of the BiDAF__ensemble_ method on SQuAD1_1 dataset for Question_Answering task?"
    main(["cpu"], query)
    # print("finished running")
    # embedding_model = sentence_transformers.SentenceTransformer(EMBED_MODEL_NAME, device=0)

