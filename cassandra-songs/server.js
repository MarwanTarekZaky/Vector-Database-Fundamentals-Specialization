const express = require('express');
const { vectorizeText, connectToCassandra, closeCassandraConnection, CASSANDRA_KEYSPACE, CASSANDRA_TABLE } = require('./util');

const app = express();
app.use(express.json());

app.post('/search', async (req, res) => {
    const text = req.body.text;
    if (!text) {
        return res.status(400).send({ error: 'Text is required' });
    }

    try {
        // use the utility function to connect to the database.
        const client = await connectToCassandra();

        // create vectorize the database using the utility code.
        const queryVector = await vectorizeText(text);

        // create the query for the database using the embedding column to search using the cosine similarity.
        // we are ordering by similarity and only asking for the first three results.
        const query = `SELECT title, similarity_cosine(embedding,  [${queryVector}]) as similarity FROM ${CASSANDRA_KEYSPACE}.${CASSANDRA_TABLE} ORDER BY embedding ANN OF [${queryVector}] LIMIT 3`;

        const result = await client.execute(query);

        if (result.rowLength > 0) {
            // map through all rows to construct a response array
            const matches = result.rows.map(row => ({
                title: row['title'],
                similarity: row['similarity']
            }));
            // return HTTP OK with the matches.
            res.status(200).send(matches);
        } else {
            // return 200 with message saying nothing was found.
            res.status(200).send({ message: 'No similar sentences found.' });
        }

    } catch (error) {
        console.error('Error:', error);
        // send back 500 if something goes wrong
        res.status(500).send({ error: 'Internal server error' });
    } finally {
        // finally, close the database connection using the method defined in the utility file.
        await closeCassandraConnection();
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});