import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from sentence_transformers import SentenceTransformer
import os
import string
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import nltk

nltk.download('punkt')
nltk.download('stopwords')

st.title('Patent Search Engine')
st.write('This search engine retrieves patents from iprsearch.ipindia.gov.in')

# Frame an input box for the user to enter the search query
search_query = st.text_input('Enter your search query here')

# Path to Patent_dataset folder
dataset_folder = 'Patent_dataset'

def load_json_from_dataset(application_no):
    """Load the theoretical abstract from the Patent_dataset folder."""
    application_no= application_no.strip()
    file_path = os.path.join(dataset_folder, f'patent_{application_no.strip()}.json')
    try:
        with open(file_path, 'r') as f:
            patent_data = json.load(f)
        return patent_data[f'{application_no}']['Abstract']
    except FileNotFoundError:
        return 'Patent file not found'
    except json.JSONDecodeError:
        return 'Error decoding JSON file'

def Semantic_Search(query):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Function to load embeddings from the JSON file
    def load_embeddings(file_path):
        try:
            with open(file_path, 'r') as f:
                embedded_data = json.load(f)
        except FileNotFoundError:
            return [], {}
        except json.JSONDecodeError:
            return [], {}

        # Load embeddings under 'Abstract' and 'Description'
        abstract_embeddings = [np.array(embedding) for embedding in embedded_data.get('Abstract', [])]
        description_embeddings = {key: np.array(value) for key, value in embedded_data.get('Description', {}).items()}

        return abstract_embeddings, description_embeddings

    # Function to perform similarity search
    def similarity_search(query, embedding_file):
        abstract_embeddings, description_embeddings = load_embeddings(embedding_file)
        if not abstract_embeddings and not description_embeddings:
            return np.array([]), []

        # Encode the query
        query_embedding = model.encode(query)

        # Calculate cosine similarity between the query and all abstracts
        similarities = []
        for embedding in abstract_embeddings:
            cos_sim = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append(cos_sim)

        # Cosine similarity for descriptions
        for embedding in description_embeddings.values():
            cos_sim = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append(cos_sim)

        combined_similarities = similarities[:len(abstract_embeddings)] + similarities[len(abstract_embeddings):]
        top5_indices = np.argsort(combined_similarities)[::-1][:5]

        return top5_indices, combined_similarities

    def get_top_similar_documents(query, application_numbers_file):
        try:
            with open(application_numbers_file, 'r') as f:
                application_numbers = f.read().splitlines()
        except FileNotFoundError:
            return []

        results = []
        for app_num in application_numbers:
            file_path = f"Patent_embeddings/patent_{app_num.strip()}_embedded.json"
            if os.path.exists(file_path):
                top5_indices, similarities = similarity_search(query, file_path)
                if top5_indices.size > 0:
                    top_similarity = similarities[top5_indices[0]]
                    results.append((app_num, top_similarity))
        
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            top5_app_numbers = [result[0] for result in results[:5]]
            top5_similarities = [result[1] for result in results[:5]]
            return top5_app_numbers, top5_similarities
        return [], []

    query = query
    application_numbers_file = "application_numbers.txt"
    top5_applications, top5_similarities = get_top_similar_documents(query, application_numbers_file)

    return top5_applications, top5_similarities

def Keyword_Search(query):
    fields_keywords = {
        "Computer Science": ["computer", "neural network", "algorithm", "machine learning", "AI", "software"],
        "Telecommunication": ["network", "signal", "wireless", "IoT", "communication"],
        "Biotechnology": ["biological", "gene", "medical", "health", "DNA", "biotech"],
        "Electronics": ["circuit", "sensor", "electronics", "microchip", "transistor"],
        "Mechanical": ["motor", "mechanical", "machine", "engine", "gear"],
    }

    def preprocess(text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        return tokens

    def detect_field_of_invention(text):
        tokens = preprocess(text)
        field_scores = {field: 0 for field in fields_keywords}

        for token in tokens:
            for field, keywords in fields_keywords.items():
                if token in keywords:
                    field_scores[field] += 1

        return max(field_scores, key=field_scores.get) if any(field_scores.values()) else None

    def extract_text_from_json(data):
        fields = []
        abstract = ""

        if "Invention Title" in data:
            fields.append(data["Invention Title"])
        if "Abstract" in data:
            abstract = data["Abstract"]
            fields.append(abstract)
        if "Description" in data and isinstance(data["Description"], list):
            fields.append(" ".join(data["Description"]))

        return " ".join(fields), abstract

    def create_inverted_index(folder_path):
        inverted_index = defaultdict(list)
        documents = []
        file_names = []
        abstracts = []
        fields_of_invention = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                        for patent_key, patent_data in data.items():
                            text, abstract = extract_text_from_json(patent_data)
                            field_of_invention = patent_data.get("Field Of Invention", "").strip()
                            if not field_of_invention:
                                field_of_invention = detect_field_of_invention(text)

                            if text.strip():
                                documents.append(text)
                                abstracts.append(abstract)
                                file_names.append(filename)
                                fields_of_invention.append(field_of_invention)

                                tokens = preprocess(text)
                                for token in set(tokens):
                                    inverted_index[token].append(filename)
                            else:
                                print(f"Skipping {filename}: No relevant text found.")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

        return inverted_index, documents, file_names, abstracts, fields_of_invention

    json_folder_path = 'Patent_dataset'
    inverted_index, documents, file_names, abstracts, fields_of_invention = create_inverted_index(json_folder_path)
    tokenized_documents = [preprocess(doc) for doc in documents]

    if len(tokenized_documents) == 0:
        print("No valid documents found. Please check your dataset.")
    else:
        bm25 = BM25Okapi(tokenized_documents)

    def search_bm25(query, top_n=5):
        query_tokens = preprocess(query)
        matching_files = set()
        for token in query_tokens:
            if token in inverted_index:
                matching_files.update(inverted_index[token])

        if not matching_files:
            print("No matching files found for the query.")
            return

        scores = bm25.get_scores(query_tokens)
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        return file_names, scores, fields_of_invention, abstracts, top_n_indices

    search_query = query
    return search_bm25(search_query, top_n=10)

if st.button('Search'):
    with st.spinner('Thinking... Please wait while we search through the patents...'):
        top5_applications, top5_similarities = Semantic_Search(search_query)
        file_names, scores, fields_of_invention, abstracts, top_n_indices = Keyword_Search(search_query)

    st.subheader('Results from Semantic Search:')
    for i, app in enumerate(top5_applications):
        abstract = load_json_from_dataset(app)
        st.write(f"**Application No:** {app.strip()}")
        st.write(f"**Cosine Similarity:** <span style='color:green'>{top5_similarities[i]:.4f}</span>", unsafe_allow_html=True)
        st.write(f"**Abstract:** {abstract}")

    st.subheader('Results from Keyword Search:')
    for i in top_n_indices:
        if scores[i] > 0:
            st.write(f"**Application No:** {file_names[i]}")
            st.write(f"**BM25 Score:** <span style='color:green'>{scores[i]:.4f}</span>", unsafe_allow_html=True)
            st.write(f"**Field of Invention:** {fields_of_invention[i]}")
            st.write(f"**Abstract:** {abstracts[i]}")
