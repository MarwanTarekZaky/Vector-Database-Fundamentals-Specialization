require('dotenv').config();
const cassandra = require('cassandra-driver');
const TimeUuid = cassandra.types.TimeUuid;
require("@tensorflow/tfjs-node");
const us_encoder = require("@tensorflow-models/universal-sentence-encoder");

let client;
let model;

// function to connect to the database and return a client.
async function connectToCassandra() {
    client = new cassandra.Client({
        contactPoints: [process.env.CASSANDRA_HOST],
        port: process.env.CASSANDRA_PORT,
        localDataCenter: 'datacenter1',
		authProvider: new cassandra.auth.PlainTextAuthProvider(process.env.CASSANDRA_USER, process.env.CASSANDRA_PASS)
    });

    await client.connect();
    console.log('Connected to Cassandra');
    return client;
}

// method that ensures there is a keyspace and table created in the database.
async function ensureKeyspaceAndTable() {
    console.log(`Ensuring KEYSPACE  ${process.env.CASSANDRA_KEYSPACE} and TABLE ${process.env.CASSANDRA_TABLE} exists ...`);
    const keyspaceQuery = `CREATE KEYSPACE IF NOT EXISTS ${process.env.CASSANDRA_KEYSPACE} WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};`;
    await client.execute(keyspaceQuery);

    const tableQuery = `CREATE TABLE IF NOT EXISTS ${process.env.CASSANDRA_KEYSPACE}.${process.env.CASSANDRA_TABLE} (
        id timeuuid PRIMARY KEY,
        artist text,
        title text,
        year int,
        sales float,
        streams float,
        downloads float,
        radio_plays float,
        rating float,
        embedding VECTOR <FLOAT, 512>
    );`;
    await client.execute(tableQuery);
    console.log(`KEYSPACE  ${process.env.CASSANDRA_KEYSPACE} and TABLE ${process.env.CASSANDRA_TABLE} created`);

    // Ensure the vector index exists for efficient vector search
    const indexQuery = `CREATE INDEX IF NOT EXISTS ann_index on ${process.env.CASSANDRA_KEYSPACE}.${process.env.CASSANDRA_TABLE}(embedding) USING 'sai';`;
    await client.execute(indexQuery);
    console.log('index created');
}

// we cache the model so we don't have to fetch it every time
async function loadModel() {
    if (!model) {
        model = await us_encoder.load();
    }
}

async function vectorizeText(text) {
    await loadModel();
    const embeddings = await model.embed([text]);
    return embeddings.arraySync()[0]; // return the first embedding
}

async function closeCassandraConnection() {
    if (client) {
        await client.shutdown();
        console.log('Cassandra connection closed');
    }
}

// export all functions and variables so other files can use it.
module.exports = {
    connectToCassandra,
    ensureKeyspaceAndTable,
    vectorizeText,
    closeCassandraConnection,
    TimeUuid,
    CASSANDRA_KEYSPACE: process.env.CASSANDRA_KEYSPACE,
    CASSANDRA_TABLE: process.env.CASSANDRA_TABLE
};