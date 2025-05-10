from src.utils.helpers import load_data
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import hnswlib
import pickle
from src.utils.helpers import load_or_compute

# #initialize the Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    
# def build_hnsw_index(embeddings: np.ndarray, ef: int = 50, M: int = 16) -> hnswlib.Index:
#     """
#     Builds and returns a HNSW index for the provided embeddings.
#     """
#     dim = embeddings.shape[1]
#     index = hnswlib.Index(space='cosine', dim=dim)
#     index.init_index(max_elements=len(embeddings), ef_construction=200, M=M)
#     index.add_items(embeddings, ids=np.arange(len(embeddings)))
#     index.set_ef(ef)
#     return index
    
# # === GLOBAL preload of data and index ===
# df = get_data()
# embeddings = np.vstack(df['excerpts_embedding'].values)
# index = build_hnsw_index(embeddings)

# Unpickling the DataFrame and HNSW index from the file
#with open('dataset_with_hnsw_index.pkl','rb') as f:
    #df , index = pickle.load(f)
    
df,index = load_or_compute()
    
# === Main query function ===
def get_similar_responses(excerpt: str, top_k: int = 5) -> list:
    print("We are in similar responses")
    query_vec = model.encode([excerpt])[0]
    ids, distances = index.knn_query(query_vec, k=top_k)
    results = []
    # return [
    #     {
    #         "prompt": prompt,
    #         "excerpt": "",
    #         "response": "",
    #         "score": 1,
    #     }, 
    #     {
    #         "prompt": prompt + "2",
    #         "excerpt": "",
    #         "response": "the second",
    #         "score": 2,
    #     }, 
    # ]

    # results = [prompt]
    for idx, dist in zip(ids[0], distances[0]):
        row = df.iloc[idx]
        results.append({
            "excerpt": row['wikipedia_excerpt'],
            #"excerpt": row.get('excerpt', ''),
            #"response": row.get('response', ''),
            "score": round(1 - dist, 4)
        })
    return results
    
"""def get_similar_responses(prompt: str, df: pd.DataFrame, index: hnswlib.Index, top_k: int = 5) -> list:

    #Given a prompt, returns top-k similar rows from the dataset.
    
    query_vec = model.encode([prompt])[0]
    ids, distances = index.knn_query(query_vec, k=top_k)

    results = []
    for idx, dist in zip(ids[0], distances[0]):
        row = df.iloc[idx]
        results.append({
            "prompt": row['prompt'],
            "excerpt": row.get('excerpt', ''),
            "response": row.get('response', ''),
            "score": round(1 - dist, 4)
        })
    return results

    
def get_similar_responses(prompt: str) -> list:
    # TODO: Implement the logic to get the similar responses
    
    #RAG response pipeline
    if df is None:
        print("[ERROR] Data not loaded")
        return []
        
    #step1: convert prompt to embedding
    prompt_embedding = prompt_to_embedding(prompt)
    #create embedding for prompt column in the dataframe
    prompt_embeddings = np.array([prompt_to_embedding(p) for p in df['prompt']])
    #compute cosine similarities between the input prompt and each prompt in the dataframe
    similarities = cosine_similarity([prompt_embedding],prompt_embeddings)
    
    
    #step2 : compute similarity of question to knowledge base hnsw
    #step3 : prune top k
    #step4: get the row text
    #step5: format the output
    
    return ["These are test responses"]
"""
