import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re  # for regular expressions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm  # For progress bars
import random

# Download required NLTK data (run this once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# 1. Load the Kaggle dataset
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)  # Replace with your dataset file

# 2. Identify the text column
text_column = "headline"  # Replace with the name of the text column in your dataset

# 3. Data Cleaning (example)
def clean_text(text):
    if isinstance(text, str):  # Check if it's a string
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters
        text = text.lower() # Lowercase
        return text
    else:
        return ""  # or some other default value

tqdm.pandas(desc="Cleaning Text")  # Initialize tqdm for Pandas
df['cleaned_text'] = df[text_column].progress_apply(clean_text)

# 4. Preprocessing (Tokenization, Stop word removal, Lemmatization)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, str):
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)
    else:
        return ""

tqdm.pandas(desc="Preprocessing Text")  # Initialize tqdm for Pandas
df['preprocessed_text'] = df['cleaned_text'].progress_apply(preprocess_text)

# 5. Embedding Generation
model_name = 'all-mpnet-base-v2'  # Or try 'all-MiniLM-L6-v2' for faster, less accurate
model = SentenceTransformer(model_name)

tqdm.pandas(desc="Generating Embeddings") # Initialize tqdm for pandas
documents = df['preprocessed_text'].tolist()  # Use the preprocessed text
document_embeddings = model.encode(documents, show_progress_bar=True) # use show_progress_bar for tqdm

# 6. Similarity Search (using FAISS)

embedding_dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(document_embeddings)

# 7. Search Function (modified to return the original document)
def search(query, index, model, df, k=3): # Add dataframe to function
    query_embedding = model.encode(query)
    query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')

    D, I = index.search(query_embedding, k) #D: distances, I: indices

    results = []
    for i in range(len(I[0])):
        document_index = I[0][i]
        results.append({
            'document': df[text_column].iloc[document_index], # Fetch original document
            'similarity': D[0][i],
            'index': document_index # Useful for retrieving other columns from dataframe
        })
    return results

# 8. Evaluation (Basic)

def evaluate(search_function, df, index, model, num_queries=5):
    """Evaluates the search function with a few random queries from the dataset."""
    random.seed(42) # Make it reproducable
    sample_indices = random.sample(range(len(df)), num_queries) #select random headlines to use as queries

    total_relevant_found = 0 # count how many relevant results are in the top-k returned

    for index_query in sample_indices:
        query = df[text_column].iloc[index_query] # use the headlines to creat queries
        results = search_function(query, index, model, df) #get search results

        print(f"Query: {query}") # print the query being used
        print("Results:")

        for result in results: # print the search results
            print(f"  Document: {result['document']}")
            print(f"  Similarity Score: {result['similarity']}")

            # VERY BASIC RELEVANCE CHECK: Does the search result have same category as the query?
            # Improve this with human evaluation for real results
            query_category = df['category'].iloc[index_query]
            result_index = result['index']
            result_category = df['category'].iloc[result_index]

            if query_category == result_category:
                print("   RELEVANT")
                total_relevant_found += 1

            else:
                print("   NOT RELEVANT")


    # Calculate a simple "relevance" metric (can be very biased)
    average_relevance = total_relevant_found / (num_queries * 3) #num_queries * k
    print(f"\nAverage Relevance (same category) in top 3 results: {average_relevance}")

    #return this to show a percentage of relevant results
    return average_relevance


# Example Usage:

# Perform a search
query = "What is the main topic?"
results = search(query, index, model, df)

print("Search Results:")
for result in results:
    print(f"Document: {result['document']}")
    print(f"Similarity Score: {result['similarity']}")


# Evaluation: Run a basic evaluation (adjust num_queries for more thoroughness)
average_relevance_score = evaluate(search, df, index, model, num_queries=5)

# Clean up memory
del model
del document_embeddings
del index