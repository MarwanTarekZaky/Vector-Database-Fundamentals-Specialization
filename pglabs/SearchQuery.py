import psycopg2
import tensorflow_hub as hub
import numpy as np

# PostgreSQL configuration
config = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'marwantarekzaky',
    'password': '', # Replace the password with the password of your postgres instance
    'port': 5432
}

# Connect to PostgreSQL database
conn = psycopg2.connect(**config)
cur = conn.cursor()

# Function to calculate cosine similarity
def check_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

# Function to search for the closest match
def search_a_match(model, query_str):
    new_vec = model([query_str])[0]
    cur.execute('SELECT * FROM mysentences')
    rows = cur.fetchall()
    match_dist = {}
    for row in rows:
        match_dist[row[1]] = check_similarity(row[2], new_vec)
    sorted_matches = sorted(match_dist.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches[0][0]

# Main function
def main():
    # Load Universal Sentence Encoder model
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Model loaded successfully")
    result = search_a_match(model, "How do I upgrade watsonx services?")
    print(f'The closest match is "{result}"')
    conn.close()

if __name__ == "__main__":
    main()
