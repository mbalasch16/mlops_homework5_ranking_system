# Helper functions can live hereimport pandas as pd
# src/utils/data_loader.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import hnswlib
import pickle

#initialize the Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
folder_path = 'pickle_files/'
pickle_file_path = os.path.join(folder_path,'dataset_with_embedding_and_index.pkl')


def load_data():
    # Get the absolute path of the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the relative path to the CSV file
    csv_path = os.path.join(base_dir, "..", "data", "6000_all_categories_questions_with_excerpts.csv")
    
    print(f"[INFO] Looking for CSV file at: {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print("[INFO] Data loaded successfully.")
        # Validate required column 'prompt'
        assert 'wikipedia_excerpt' in df.columns, "'wikipedia_excerpt' column is missing in CSV."
    
        # Step 1: Convert prompt to embedding
        print("Converting wikipedia_excerpt to embeddings...")
        df['excerpts_embedding'] = df['wikipedia_excerpt'].apply(lambda x: model.encode([x])[0])
        return df
    except FileNotFoundError:
        print(f"[ERROR] Could not find CSV file at: {csv_path}")
        raise  # Raise the error to be handled by the caller
    return df
    

# #global variable to store loaded dataset
# df = None

# def get_data():
#     #calling load_file function
#     global df
#     df = load_data()
#     print("Data is loaded")
    
#     # Validate required column 'prompt'
#     assert 'wikipedia_excerpt' in df.columns, "'wikipedia_excerpt' column is missing in CSV."
    
#     # Step 1: Convert prompt to embedding
#     print("Converting wikipedia_excerpt to embeddings...")
#     df['excerpts_embedding'] = df['wikipedia_excerpt'].apply(lambda x: model.encode([x])[0])
    
#     return df
    
def build_hnsw_index(embeddings: np.ndarray, ef: int = 50, M: int = 16) -> hnswlib.Index:
    """
    Builds and returns a HNSW index for the provided embeddings.
    """
    dim = embeddings.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(embeddings), ef_construction=200, M=M)
    index.add_items(embeddings, ids=np.arange(len(embeddings)))
    index.set_ef(ef)
    return index
    
# # === GLOBAL preload of data and index ===
# df = load_data()
# embeddings = np.vstack(df['excerpts_embedding'].values)
# index = build_hnsw_index(embeddings)




def load_or_compute():
    """
    Loads the data and HNSW index from pickle if available. If not, computes and saves them.
    """
    # Check if the pickle file exists
    if os.path.exists(pickle_file_path):
        print("[INFO] Loading data and index from pickle file...")
        with open(pickle_file_path, 'rb') as f:
            df, index = pickle.load(f)
        print("[INFO] Data and index loaded successfully.")
    else:
        print("[INFO] Pickle file not found. Computing embeddings and index...")
        # === GLOBAL preload of data and index ===
        df = load_data()
        embeddings = np.vstack(df['excerpts_embedding'].values)
        index = build_hnsw_index(embeddings)

        # Save the DataFrame and index to a pickle file
        with open(pickle_file_path, 'wb') as f:
            pickle.dump((df, index), f)
        print(f"[INFO] Pickle file saved at: {pickle_file_path}")

    return df, index


# Now df contains the DataFrame with embeddings and index is the HNSW index you can use for similarity search.

# with open(pickle_file_path,'wb') as f:
#     pickle.dump((df,index),f)
    
# print("DataFrame with embeddings and HNSW index has been pickled and saved.")


