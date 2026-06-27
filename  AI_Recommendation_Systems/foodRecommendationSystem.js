import "dotenv/config";
import { HfInference } from "@huggingface/inference";
import foodItems from "./FoodDataSet.js";

const hfToken = process.env.HF_TOKEN;
if (!hfToken) {
  throw new Error("Missing HF_TOKEN environment variable.");
}

const hf = new HfInference(hfToken);
const collectionName = "food_collection";
const chromaBaseUrl = "http://localhost:8000/api/v2";
const chromaTenant = "default_tenant";
const chromaDatabase = "default_database";

async function chromaRequest(path, options = {}) {
  const response = await fetch(`${chromaBaseUrl}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const message = data?.message || data?.error || response.statusText;
    throw new Error(`Chroma request failed (${response.status}): ${message}`);
  }

  return data;
}

async function getOrCreateCollection(name) {
  return chromaRequest(
    `/tenants/${chromaTenant}/databases/${chromaDatabase}/collections`,
    {
      method: "POST",
      body: JSON.stringify({
        name,
        get_or_create: true,
      }),
    },
  );
}

async function upsertCollectionRecords(collectionId, payload) {
  await chromaRequest(
    `/tenants/${chromaTenant}/databases/${chromaDatabase}/collections/${collectionId}/upsert`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

async function getCollectionCount(collectionId) {
  return chromaRequest(
    `/tenants/${chromaTenant}/databases/${chromaDatabase}/collections/${collectionId}/count`,
    {
      method: "GET",
    },
  );
}

async function queryCollection(collectionId, payload) {
  return chromaRequest(
    `/tenants/${chromaTenant}/databases/${chromaDatabase}/collections/${collectionId}/query`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

async function main() {
  try {
    console.log("Connecting to Chroma...");
    const collection = await getOrCreateCollection(collectionName);
    const uniqueIds = new Set();
    foodItems.forEach((food, index) => {
      while (uniqueIds.has(food.food_id.toString())) {
        food.food_id = `${food.food_id}_${index}`;
      }
      uniqueIds.add(food.food_id.toString());
    });

    const expectedRecordCount = foodItems.length;
    const existingRecordCount = await getCollectionCount(collection.id);
    const foodTexts = foodItems.map(
      (food) => `${food.food_name}. ${food.food_description}. Ingredients: ${food.food_ingredients.join(", ")}`,
    );

    if (existingRecordCount !== expectedRecordCount) {
      console.log(`Indexing ${expectedRecordCount} food items in Chroma...`);
      const embeddingsData = await generateEmbeddings(foodTexts);
      await upsertCollectionRecords(collection.id, {
        ids: foodItems.map((food) => food.food_id.toString()),
        documents: foodTexts,
        embeddings: embeddingsData,
        metadatas: foodItems.map((food) => ({
          cuisine_type: food.cuisine_type.toLowerCase(),
          cooking_method: food.cooking_method.toLowerCase(),
        })),
      });
    } else {
      console.log(`Reusing existing Chroma collection with ${existingRecordCount} indexed food items.`);
    }

    const query = "I want to eat vegan food";
    console.log(`Searching for: ${query}`);
    const filterCriteria = await extractFilterCriteria(query);
    const initialResults = await performSimilaritySearch(collection, query, filterCriteria);
    initialResults.slice(0, 5).forEach((item, index) => {
      console.log(`Top ${index + 1} Recommended Food Name ==>, ${item.food_name}`);
    });
  } catch (error) {
    console.error("Error in main function:", error);
  }
}

async function generateEmbeddings(texts) {
  const results = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L12-v2",
    inputs: texts,
  });
  return results;
}

async function classifyText(text, labels) {
    const responses = await hf.zeroShotClassification({
      model: "facebook/bart-large-mnli",
      inputs: text,
      parameters: {
        candidate_labels: labels,
      },
    });
    return responses;
  }

async function extractFilterCriteria(query) {
  const criteria = { diet: null, cuisine: null };
  const normalizedQuery = query.toLowerCase();

  const dietLabels = ["vegan", "non-vegan", "vegetarian", "non-vegetarian", "pescatarian", "omnivore", "paleo", "ketogenic"];
  const cuisineLabels = ["chinese", "indian", "japanese"];

  const matchedDietLabel = dietLabels.find((label) => normalizedQuery.includes(label));
  if (matchedDietLabel) {
    criteria.diet = matchedDietLabel;
  }

  const matchedCuisineLabel = cuisineLabels.find((label) => normalizedQuery.includes(label));
  if (matchedCuisineLabel) {
    criteria.cuisine = matchedCuisineLabel;
  }

  if (criteria.diet || criteria.cuisine) {
    console.log("Extracted Filter Criteria:", criteria);
    return criteria;
  }

  const dietResult = await classifyText(query, dietLabels);
  const highestDietMatch = dietResult[0];
  const highestDietScoreLabel = highestDietMatch?.label;
  const dietScore = highestDietMatch?.score ?? 0;

// Only apply diet criteria if the score is very high (e.g., > 0.8)
  if (dietScore > 0.8) {
    criteria.diet = highestDietScoreLabel;
  } else {
    const cuisineResult = await classifyText(query, cuisineLabels);
    const highestCuisineMatch = cuisineResult[0];
    const highestCuisineScoreLabel = highestCuisineMatch?.label;
    const cuisineScore = highestCuisineMatch?.score ?? 0;

 // Only apply cuisine criteria if the score is very high (e.g., > 0.8)
    if (cuisineScore > 0.8) {
      criteria.cuisine = highestCuisineScoreLabel;
    }
  }
  console.log('Extracted Filter Criteria:', criteria);
  return criteria;
}

async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
  try {
    const queryEmbedding = await generateEmbeddings([queryTerm]);
    console.log(filterCriteria);
    const where = filterCriteria.cuisine
      ? { cuisine_type: filterCriteria.cuisine }
      : undefined;
    const results = await queryCollection(collection.id, {
      query_embeddings: queryEmbedding,
      n_results: 5,
      include: ["distances", "documents", "metadatas"],
      where,
    });

    if (!results?.ids?.[0]?.length) {
      console.log(`No food items found similar to "${queryTerm}"`);
       return [];
     }

      let topFoodItems = results.ids[0].map((id, index) => {
        return {
          id,
           score: results.distances[0][index],
            food_name: foodItems.find(item => item.food_id.toString() === id).food_name,
             food_description: foodItems.find(item => item.food_id.toString() === id).food_description
         };
    }).filter(Boolean);
    return topFoodItems.sort((a, b) => a.score - b.score);
  } catch (error) {
    console.error("Error during similarity search:", error);
    return [];
  }
}

main();
