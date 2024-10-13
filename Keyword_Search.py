import os
import json
import string
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import nltk

# Download NLTK stopwords and punkt tokenizer...Run only once
nltk.download('punkt')
nltk.download('stopwords')

# Define keyword lists for fields of invention
fields_keywords = {
    "Computer Science": ["computer", "neural network", "algorithm", "machine learning", "AI", "software"],
    "Telecommunication": ["network", "signal", "wireless", "IoT", "communication"],
    "Biotechnology": ["biological", "gene", "medical", "health", "DNA", "biotech"],
    "Electronics": ["circuit", "sensor", "electronics", "microchip", "transistor"],
    "Mechanical": ["motor", "mechanical", "machine", "engine", "gear"],
    # Add more fields and keywords as necessary
}

# Function to preprocess text: tokenize, lowercase, and remove stopwords and punctuation
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

# Function to detect the field of invention based on keywords in title, description, or abstract
def detect_field_of_invention(text):
    tokens = preprocess(text)
    field_scores = {field: 0 for field in fields_keywords}  # Initialize field scores

    for token in tokens:
        for field, keywords in fields_keywords.items():
            if token in keywords:
                field_scores[field] += 1  # Increment score if keyword is found

    # Return the field with the highest score or None if no significant match
    return max(field_scores, key=field_scores.get) if any(field_scores.values()) else None

# Function to extract relevant fields and combine them into a single text
def extract_text_from_json(data):
    fields = []
    abstract = ""
    
    # Extract text fields from the nested JSON
    if "Invention Title" in data:
        fields.append(data["Invention Title"])
    if "Abstract" in data:
        abstract = data["Abstract"]  # Save the Abstract separately
        fields.append(abstract)
    if "Description" in data and isinstance(data["Description"], list):
        fields.append(" ".join(data["Description"]))  # Join description list into a single string
    
    # Combine all extracted fields into one text block
    return " ".join(fields), abstract

# Function to create an inverted index and document list
def create_inverted_index(folder_path):
    inverted_index = defaultdict(list)  # Hash map to store word -> list of file names
    documents = []
    file_names = []
    abstracts = []
    fields_of_invention = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # Loop through each patent entry (the key is the patent application number)
                    for patent_key, patent_data in data.items():
                        # Extract relevant text and abstract from the JSON structure
                        text, abstract = extract_text_from_json(patent_data)
                        
                        # Use the actual "Field Of Invention" if present, otherwise fallback to keyword detection
                        field_of_invention = patent_data.get("Field Of Invention", "").strip()
                        if not field_of_invention:  # If no field, detect it using keywords
                            field_of_invention = detect_field_of_invention(text)
                        
                        # Skip files with no meaningful text
                        if text.strip():
                            documents.append(text)
                            abstracts.append(abstract)  # Append the abstract separately
                            file_names.append(filename)
                            fields_of_invention.append(field_of_invention)

                            # Tokenize the text and update the inverted index
                            tokens = preprocess(text)
                            for token in set(tokens):  # Use set to avoid duplicates
                                inverted_index[token].append(filename)
                        else:
                            print(f"Skipping {filename}: No relevant text found.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return inverted_index, documents, file_names, abstracts, fields_of_invention

# Path to the folder containing the JSON files
json_folder_path = 'Patent_dataset'

# Create inverted index, documents, file names list, abstracts, and fields of invention
inverted_index, documents, file_names, abstracts, fields_of_invention = create_inverted_index(json_folder_path)

# Preprocess documents for BM25
tokenized_documents = [preprocess(doc) for doc in documents]

# Check if there are any valid documents before initializing BM25
if len(tokenized_documents) == 0:
    print("No valid documents found. Please check your dataset.")
else:
    # Initialize BM25 model with tokenized documents
    bm25 = BM25Okapi(tokenized_documents)

# Function to search using BM25 with automatic field filtering
def search_bm25(query, top_n=5):
    # Preprocess the query
    query_tokens = preprocess(query)
    
    # Find files matching the query terms using the inverted index
    matching_files = set()
    for token in query_tokens:
        if token in inverted_index:
            matching_files.update(inverted_index[token])

    # If no matching files found
    if not matching_files:
        print("No matching files found for the query.")
        return

    # Get BM25 scores for all documents and rank them
    scores = bm25.get_scores(query_tokens)
    
    # Get the top N results based on BM25 scores
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    
    # Print the top N matching documents with their file names, scores, and the full abstract
    for index in top_n_indices:
        if scores[index] > 0:  # Skip documents with zero relevance
            print(f"File: {file_names[index]} - Score: {scores[index]}")
            print(f"Field Of Invention: {fields_of_invention[index]}")
            print(f"Abstract: {abstracts[index]}")
            print("\n")

# Example usage:
search_query = "Manga anime comics"
search_bm25(search_query, top_n=5)
