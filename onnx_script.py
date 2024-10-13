
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import json
import os
import onnxruntime as ort
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
import nltk


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load the ONNX model
ONNX_MODEL_PATH = "./model.onnx"
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH)
except Exception as e:
    print(f"Failed to load ONNX model: {e}")
    raise

def embedding(text):
    # Tokenize the entire paragraph (multiple sentences grouped together)
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs.get('token_type_ids', np.zeros_like(input_ids))

    input_names = [input.name for input in session.get_inputs()]

    # Run inference on the paragraph
    outputs = session.run(None, {
        input_names[0]: input_ids,
        input_names[1]: attention_mask,
        input_names[2]: token_type_ids
    })

    token_embeddings = outputs[0]
    embedding = np.mean(token_embeddings, axis=1).squeeze()
    
    # Convert ndarray to list for JSON serializability
    return embedding.tolist()

def Get_embedding(application_no, folder_name):
    # Load the input data (a JSON file)
    application_no = application_no.strip()
    input_file = os.path.join(os.path.dirname(__file__), f'Patent_dataset/patent_{application_no}.json')
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {input_file}")
        return
    
    # Check if the application_no exists in data
    if application_no not in data:
        print(f"Application number {application_no} not found in the data.")
        return
    
    sample_data = data[application_no]
    
    # Keep the Abstract and Description as they are, no need for sentence tokenization here
    sample_data['Abstract'] = sample_data['Abstract']
    sample_data['Description'] = sample_data['Description']
    
    df = pd.DataFrame([sample_data])
    
    # Embed the abstract as a whole paragraph using multiprocessing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(embedding, df['Abstract'].values[0])]
        abstract_embeddings = [future.result() for future in as_completed(futures)]
    
    # Embed each paragraph of the description as a whole
    description_embeddings = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(embedding, paragraph) for paragraph in df['Description'].values[0]]
        for i, future in enumerate(as_completed(futures)):
            paragraph_embedding = future.result()
            description_embeddings[f'Paragraph_{i+1}'] = paragraph_embedding
    
    # Create the output folder if it does not exist
    output_folder = os.path.join(os.path.dirname(__file__), folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save both abstract and description embeddings in the same JSON file in the output folder
    output_file = os.path.join(output_folder, f'patent_{application_no}_embedded.json')
    with open(output_file, 'w') as f:
        json.dump({
            'Abstract': abstract_embeddings,
            'Description': description_embeddings
        }, f, indent=4)

# Guard the main execution block with `if __name__ == '__main__':`
if __name__ == '__main__':
    # Read the application numbers from the file and process each one
    with open('application_numbers.txt', 'r') as f:
        application_numbers = f.read().splitlines()

    for application_no in application_numbers:
        Get_embedding(application_no, 'Patent_embeddings')
        print(f"Embedding done for Application No: {application_no}")
