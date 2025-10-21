import os
import time
import uuid
import numpy as np
import jsonlines
from concurrent.futures import ProcessPoolExecutor
import sentence_transformers
import chromadb

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHROMA_PERSIST_DIRECTORY = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/chroma_db/agenda"
CHROMA_COLLECTION_NAME = "all"
FILE_PATH = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/agenda/agenda_descriptions_merged.jsonl"

def sentence_embedding(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts)

def create_chroma_db_local(persist_directory, collection_name):
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection

def load_model_offline(model_name, cache_dir=None):
    """Load model from local cache directory for offline use"""
    if cache_dir is None:
        cache_dir = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/models/sentence_transformers"
    
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

def insert_to_db(texts, model_name, device, db, cache_dir=None):
    model = load_model_offline(model_name, cache_dir)
    batch_embeddings = []
    batch_texts = []
    start_time = time.time()
    print(f"Total Articles to process: {len(texts)}, Device: {device}.")
    for i, text in enumerate(texts):
        embedding = sentence_embedding(model, text)[0].tolist()  # Ensure it's a vector, not a batch
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
            print(f"Completed {i+1} articles on {device}, Time taken: {time.time() - start_time:.2f}s.")
    print(f"Done with device {device}. Total time: {time.time() - start_time:.2f}s.")

def query_llm(devices, query, cache_dir=None):
    input_texts = []
    db = create_chroma_db_local(CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME)

    with open(FILE_PATH, 'r') as f:
        for item in jsonlines.Reader(f):
            input_texts.append(item["event"])

    print("Total Number of Agendas:", len(input_texts))

    # 🔥 Always repopulate the DB for now (to ensure content exists)
    if len(db.get()["ids"]) == 0:
        print("DB is empty. Populating...")
        insert_to_db(input_texts, model_name=EMBED_MODEL_NAME, device="cpu", db=db, cache_dir=cache_dir)
    else:
        print("DB already populated. Skipping insert.")

    model = load_model_offline(EMBED_MODEL_NAME, "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/chroma_db/agenda")
    query_embedding = sentence_embedding(model, query).tolist()

    print("Querying DB...")
    results = db.query(query_embeddings=query_embedding, n_results=3)
    if not results["documents"] or not results["documents"][0]:
        print("No documents retrieved.")
        return "No results found."

    retrieval_content = [result for result in results['documents'][0]]
    return '\n'.join(retrieval_content)

def main(devices, query, cache_dir=None):
    result = query_llm(devices, query, cache_dir)
    print("=== Retrieved ===")
    print(result)
    download_model_for_offline_use(EMBED_MODEL_NAME, "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/chroma_db/agenda")

def download_model_for_offline_use(model_name=EMBED_MODEL_NAME, cache_dir=None):
    """Call this function on login node to download model for offline use"""
    print(f"Downloading model {model_name} for offline use...")
    load_model_offline(model_name, cache_dir)
    print("Model download complete!")

if __name__ == '__main__':
    query = "What did Adam do from 7:00 PM to 9:00 PM on 2022/10/27?"
    #query = "What is Jessica's agenda on March 7th, 2023?"
    main(["cpu"], query)