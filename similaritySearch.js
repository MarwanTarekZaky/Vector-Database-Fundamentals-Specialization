import { ChromaClient, DefaultEmbeddingFunction } from "chromadb";

// Create Chroma client (ensure server is running on localhost:8000)
const client = new ChromaClient({
  path: "http://localhost:8000"
});

// Initialize embedding function
const default_emd = new DefaultEmbeddingFunction();

// Collection name
const collectionName = "my_grocery_collection";

// Main function
async function main() {
  try {
    // Create or get collection
    const collection = await client.getOrCreateCollection({
      name: collectionName,
      embeddingFunction: default_emd
    });

    // Grocery items
    const texts = [
      'fresh red apples',
      'organic bananas',
      'ripe mangoes',
      'whole wheat bread',
      'farm-fresh eggs',
      'natural yogurt',
      'frozen vegetables',
      'grass-fed beef',
      'free-range chicken',
      'fresh salmon fillet',
      'aromatic coffee beans',
      'pure honey',
      'golden apple',
      'red fruit'
    ];

    // Generate IDs
    const ids = texts.map((_, index) => `food_${index + 1}`);

    // Generate embeddings
    const embeddingsData = await default_emd.generate(texts);

    // Add to collection
    await collection.add({
      ids,
      documents: texts,
      embeddings: embeddingsData
    });

    // Retrieve stored data
    const allItems = await collection.get({
      include: ["documents"] // (embeddings optional)
    });

    console.log("All stored documents:");
    console.log(allItems);

    // Perform similarity search
    await performSimilaritySearch(collection, allItems);

  } catch (error) {
    console.error("Error:", error);
  }
}

// Similarity search function
async function performSimilaritySearch(collection, allItems) {
  try {
    const queryTerm = "apple";

    const results = await collection.query({
      queryTexts: [queryTerm],
      nResults: 3
    });

    console.log("\nRaw query results:");
    console.log(results);

    if (!results || !results.ids || results.ids.length === 0) {
      console.log(`No documents found similar to "${queryTerm}"`);
      return;
    }

    console.log(`\nTop 3 similar documents to "${queryTerm}":`);

    for (let i = 0; i < results.ids[0].length; i++) {
      const id = results.ids[0][i];
      const score = results.distances[0][i];

      const text = allItems.documents[allItems.ids.indexOf(id)];

      console.log(` - ID: ${id}, Text: '${text}', Score: ${score}`);
    }

  } catch (error) {
    console.error("Error during similarity search:", error);
  }
}

// Run program
main();