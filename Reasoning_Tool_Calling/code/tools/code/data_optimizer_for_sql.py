import pandas as pd
import json
import re
import demoji
import os

def remove_emoji(string):
    """Removes emojis from a string."""
    if not isinstance(string, str):
        return string
    cleaned_string = demoji.replace_with_desc(string)
    return re.sub(r'[\U00010000-\U0010FFFF]', '', cleaned_string)

def preprocess_flights():
    print("Processing flights data...")
    file_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/flights/Combined_Flights_2022.csv"
    output_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/flights/flights_data.parquet"
    
    # Use chunking for large file
    chunk_iter = pd.read_csv(file_path, chunksize=100000, low_memory=False)
    all_chunks = []
    for chunk in chunk_iter:
        all_chunks.append(chunk.fillna("---"))
    
    data = pd.concat(all_chunks, ignore_index=True)
    # Convert all columns to string to prevent type inference errors with '---'
    for col in data.columns:
        data[col] = data[col].astype(str)
    data.to_parquet(output_path)
    print("Flights data saved to Parquet.")

def preprocess_coffee():
    print("Processing coffee data...")
    file_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/coffee/coffee_price.csv"
    output_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/coffee/coffee_data.parquet"
    
    data = pd.read_csv(file_path).fillna("---")
    # Convert all columns to string
    for col in data.columns:
        data[col] = data[col].astype(str)
    data.to_parquet(output_path)
    print("Coffee data saved to Parquet.")

def preprocess_airbnb():
    print("Processing Airbnb data...")
    file_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/airbnb/Airbnb_Open_Data.csv"
    output_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/airbnb/airbnb_data.parquet"

    data = pd.read_csv(file_path, low_memory=False).fillna("---")
    data.columns = [c.replace(" ","_").replace("lat", "latitude").replace("long", "longitude") for c in data.columns]
    
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].apply(remove_emoji)
        
    # Convert all columns to string to prevent type inference errors
    for col in data.columns:
        data[col] = data[col].astype(str)
        
    data.to_parquet(output_path)
    print("Airbnb data saved to Parquet.")

def preprocess_yelp():
    print("Processing Yelp data...")
    file_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/yelp/yelp_academic_dataset_business.json"
    output_path = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/external_corpus/yelp/yelp_data.parquet"
    
    with open(file_path) as f:
        yelp_json = [json.loads(line) for line in f]
    
    data = pd.DataFrame(yelp_json).fillna("---")
    # Ensure all data is string to avoid issues with Parquet
    for col in data.columns:
        data[col] = data[col].astype(str)

    data.to_parquet(output_path)
    print("Yelp data saved to Parquet.")

def main():
    preprocess_flights()
    preprocess_coffee()
    preprocess_airbnb()
    preprocess_yelp()
    print("\nAll data has been pre-processed and saved to Parquet files.")

if __name__ == "__main__":
    main()