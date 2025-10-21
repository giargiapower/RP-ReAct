from ollama import Client

def download_qwen_model():
    client = Client()
    
    # Download Qwen2.5 14B model
    try:
        print("Downloading Qwen2.5 14B model...")
        client.pull("qwen3:14b")
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_qwen_model()

