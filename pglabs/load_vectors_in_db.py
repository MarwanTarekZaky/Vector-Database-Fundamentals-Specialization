import psycopg2
from psycopg2 import Error
import tensorflow_hub as hub
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Model loaded successfully")

# PostgreSQL database configuration
config = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'marwantarekzaky',
    'password': '', # Replace the password with the password of your postgres instance
    'port': 5432
}

# Sample sentences
sentences = ['Go to Try IBM watsonx.ai or Try watsonx.governance. If you sign up for watsonx.governance, you automatically provision watsonx.ai as well.',
                                  'Yes, when you sign up for IBM watsonx.ai, you automatically provision the free version of the underlying services: Watson Studio, Watson Machine Learning, and IBM Cloud Object Storage. When you sign up for IBM watsonx.governance, you automatically provision the free version of Watson OpenScale and the free versions of the services for IBM watsonx.ai.',
                                  'When you are ready to upgrade any of the underlying services for watsonx.ai or watsonx.governance, you can upgrade in place without losing any of your work or data. You must be the owner or administrator of the IBM Cloud account for a service to upgrade it. See Upgrading services on watsonx.'
]

def vectorize(sentence):
    return model([sentence])[0].numpy().tolist()

def add_data():
    try:
        # Connect to PostgreSQL database
        connection = psycopg2.connect(**config)

        # Create a cursor object
        cursor = connection.cursor()

        # Define SQL query to create table
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS mysentences (
                id SERIAL PRIMARY KEY,
                sentence TEXT,
                embedding FLOAT[]
            );
        '''

        # Execute the query to create the table
        cursor.execute(create_table_query)
        connection.commit()
        print('Table created successfully')

        # Insert data into the table
        for sentence in sentences:
            embeddings = vectorize(sentence)
            insert_query = '''
                INSERT INTO mysentences (sentence, embedding)
                VALUES (%s, %s);
            '''
            cursor.execute(insert_query, (sentence, embeddings))
            connection.commit()

        # Fetch and display the data
        select_query = "SELECT * FROM mysentences;"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        for row in rows:
            print(row[1], end='\n\n')  # Print the 'sentence' column

    except Error as e:
        print(f"Error connecting to PostgreSQL: {e}")

    finally:
        # Close the connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

if __name__ == "__main__":
    add_data()