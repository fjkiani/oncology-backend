import chromadb
import pprint

CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "clinical_trials_eligibility"

print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    print(f"Getting collection: {CHROMA_COLLECTION_NAME}")
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    # 1. Check Count
    count = collection.count()
    print(f"Collection count: {count}")

    if count > 0:
        # 2. Get a sample item (replace with a real source_url from your data if known)
        # Let's try getting the first few items instead if we don't know a specific URL
        print("\nGetting first 2 items (if available):")
        results = collection.get(
            limit=2,
            include=['metadatas', 'documents', 'embeddings'] # Request embeddings explicitly
        )
        pprint.pprint(results)

        # Optionally check embedding dimension of the first result if it exists
        if results and results.get('embeddings') and results['embeddings']:
             print(f"Dimension of first embedding vector: {len(results['embeddings'][0])}")
        else:
             print("Could not retrieve embeddings for sample.")

    else:
        print("Collection is empty.")

except Exception as e:
    print(f"An error occurred: {e}")

print("\nVerification complete.")
