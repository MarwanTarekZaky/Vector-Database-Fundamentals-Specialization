
import { ChromaClient, DefaultEmbeddingFunction } from "chromadb";

const client = new ChromaClient();
const default_emd = new DefaultEmbeddingFunction();
const collectionName = "my_flower_collection";
async function main() {
    try {
        const collection = await client.getOrCreateCollection({
            name: collectionName,
            embeddings: default_emd
        });
        const flowers = [
            "A vibrant red rose",
            "A sunny yellow tulip",
            "A pure white lily",
        ];
        const ids = flowers.map((_, index) => `flower_${index + 1}`);
        const embeddingsData = await default_emd.generate(flowers); // Generate embeddings
        await collection.add({ ids, documents: flowers, embeddings: embeddingsData });
        const allItems = await collection.get(); // Retrieve all items
        console.log(allItems);
        await performSimilaritySearch(collection, allItems);
    } catch (error) {
        console.error("Error:", error);
    }
}
async function performSimilaritySearch(collection, allItems) {
    try {
        const queryTerm = "yellow flowers";
        const results = await collection.query({
            collection: collectionName,
            queryTexts: [queryTerm],
            n: 3
        });
        if (!results || !results.ids || results.ids.length === 0) {
            console.log(`No documents found similar to "${queryTerm}"`);
            return;
        }
        console.log(`Top 3 similar documents to "${queryTerm}":`);
        for (let i = 0; i < results.ids[0].length; i++) {
            const id = results.ids[0][i];
            const score = results.distances[0][i];
            const text = allItems.documents[allItems.ids.indexOf(id)];
            console.log(` - ID: ${id}, Text: '${text}', Score: ${score}`);
        }
    } catch (error) {
        console.error('Error during similarity search:', error);
    }
}
main();