import { ChromaClient } from "chromadb";

const client = new ChromaClient({
  path: "http://localhost:8000"
});

async function main() {
  try {
    const collection = await client.getOrCreateCollection({
      name: "my_basic_collection",
    });

    const texts = [
      "This is sample text 1.",
      "This is sample text 2.",
      "This is sample text 3.",
    ];

    const ids = texts.map((_, index) => `document_${index + 1}`);

    await collection.add({ ids, documents: texts });

    const allItems = await collection.get();
    console.log(allItems);

  } catch (error) {
    console.error("Error:", error);
  }
}

main();