import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from sentence_transformers import SentenceTransformer
import os

# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to load embeddings from the JSON file
def load_embeddings(file_path):
    try:
        with open(file_path, 'r') as f:
            embedded_data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return [], {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return [], {}
    
    # Load embeddings under 'Abstract' and 'Description'
    abstract_embeddings = [np.array(embedding) for embedding in embedded_data.get('Abstract', [])]
    
    # For descriptions, load each paragraph's embedding
    description_embeddings = {key: np.array(value) for key, value in embedded_data.get('Description', {}).items()}
    
    return abstract_embeddings, description_embeddings

# Function to perform similarity search
def similarity_search(query, embedding_file):
    # Load the corresponding embeddings
    abstract_embeddings, description_embeddings = load_embeddings(embedding_file)
    
    if not abstract_embeddings and not description_embeddings:
        print(f"No embeddings found for file: {embedding_file}")
        return np.array([]), []  # Return an empty array if no embeddings found
    
    # Encode the query
    query_embedding = model.encode(query)
    
    # Calculate cosine similarity between the query and all abstracts
    similarities = []
    
    # Cosine similarity for abstracts
    for embedding in abstract_embeddings:
        cos_sim = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities.append(cos_sim)
    
    # Cosine similarity for descriptions (all paragraphs)
    for paragraph, embedding in description_embeddings.items():
        cos_sim = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities.append(cos_sim)
    
    # Combine both abstract and description similarities
    combined_similarities = similarities[:len(abstract_embeddings)] + similarities[len(abstract_embeddings):]
    
    # Get the indices of the top 5 most similar documents
    top5_indices = np.argsort(combined_similarities)[::-1][:5]
    
    return top5_indices, combined_similarities

# Function to get the application numbers of the top 5 most similar documents
def get_top_similar_documents(query, application_numbers_file):
    try:
        with open(application_numbers_file, 'r') as f:
            application_numbers = f.read().splitlines()
    except FileNotFoundError:
        print(f"File not found: {application_numbers_file}")
        return []
    
    # Store the best 5 results
    results = []
    
    for app_num in application_numbers:
        # Construct the file path for the corresponding embedding file
        app_num = app_num.strip()
        file_path = f"Patent_embeddings/patent_{app_num}_embedded.json"
        
        if os.path.exists(file_path):
            # Perform similarity search for each application number
            top5_indices, similarities = similarity_search(query, file_path)
            
            if top5_indices.size > 0:  # Only append if we get valid results
                # Store the application number and the similarity score of the top result
                top_similarity = similarities[top5_indices[0]]  # Best match
                results.append((app_num, top_similarity))
        else:
            print(f"Embedding file not found: {file_path}")
    
    # Sort the results based on the similarity score (for now, only considering the first in top5_indices)
    if results:
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by the best similarity score
    
        # Get top 5 application numbers (in sorted order)
        top5_app_numbers = [result[0] for result in results[:5]]
        top5_similarities = [result[1] for result in results[:5]]
    
        return top5_app_numbers, top5_similarities
    else:
        print("No valid results found.")
        return [], []

# Query to search for similar documents
query = "Anime And Manga" 
application_numbers_file = "application_numbers.txt"  # Your text file containing application numbers

top5_applications, top5_similarities = get_top_similar_documents(query, application_numbers_file)

# Print the top 5 most similar application numbers and their cosine similarities
if top5_applications:
    print("Top 5 similar application numbers with cosine similarity:")
    for app_number, similarity in zip(top5_applications, top5_similarities):
        print(f"Application No: {app_number}, Cosine Similarity: {similarity:.4f}")
else:
    print("No top 5 similar application numbers found.")
