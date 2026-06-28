import { ChromaClient } from "chromadb";
import "dotenv/config";
import { HfInference } from "@huggingface/inference";
import jobPostings from "./jobPostings.js";

const client = new ChromaClient({ path: "http://localhost:8000" });
const hf = new HfInference(process.env.HF_TOKEN);
const collectionName = "job_collection";
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
	return chromaRequest(`/tenants/${chromaTenant}/databases/${chromaDatabase}/collections`, {
		method: "POST",
		body: JSON.stringify({
			name,
			get_or_create: true,
		}),
	});
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
	const query = "Creative Studio";

	try {
		const collection = await getOrCreateCollection(collectionName);
		const filterCriteria = await extractFilterCriteria(query);
		const filteredJobPostings = filterJobPostings(jobPostings, filterCriteria);
		const uniqueIds = new Set();

		jobPostings.forEach((job, index) => {
			while (uniqueIds.has(job.jobId.toString())) {
				job.jobId = `${job.jobId}_${index}`;
			}
			uniqueIds.add(job.jobId.toString());
		});

		const jobTexts = jobPostings.map(
			(job) => `${job.jobTitle}. ${job.jobDescription}. ${job.jobType}. ${job.location}`,
		);
		const embeddingsData = await generateEmbeddings(jobTexts);

		await upsertCollectionRecords(collection.id, {
			ids: jobPostings.map((job) => job.jobId.toString()),
			documents: jobTexts,
			embeddings: embeddingsData,
		});

		const initialResults = await performSimilaritySearch(collection, query, jobPostings);
		initialResults.slice(0, 3).forEach((item, index) => {
			console.log(`Top ${index + 1} Job Title: ${item.jobTitle}`);
			console.log(`Top ${index + 1} Job Type: ${item.jobType}`);
			console.log(`Top ${index + 1} Job Description: ${item.jobDescription}`);
			console.log(`Top ${index + 1} Company: ${item.company}`);
		});

		console.log(`Collection ready: ${collection.name}`);
		console.log(`Filtered postings count: ${filteredJobPostings.length}`);
	} catch (error) {
		console.error("Error in main function:", error);
	}
}

function filterJobPostings(jobPostings, filterCriteria) {
	return jobPostings.filter((jobPosting) => {
		if (filterCriteria.location && !jobPosting.location.toLowerCase().includes(filterCriteria.location.toLowerCase())) {
			return false;
		}

		if (filterCriteria.jobTitle && !jobPosting.jobTitle.toLowerCase().includes(filterCriteria.jobTitle.toLowerCase())) {
			return false;
		}

		if (filterCriteria.jobType && !jobPosting.jobType.toLowerCase().includes(filterCriteria.jobType.toLowerCase())) {
			return false;
		}

		if (filterCriteria.company && !jobPosting.company.toLowerCase().includes(filterCriteria.company.toLowerCase())) {
			return false;
		}

		return true;
	});
}

async function generateEmbeddings(texts) {
	const results = await hf.featureExtraction({
		model: "sentence-transformers/all-MiniLM-L6-v2",
		inputs: texts,
	});
	return results;
}

async function classifyText(word, labels) {
	const response = await hf.zeroShotClassification({
		model: "facebook/bart-large-mnli",
		inputs: word,
		parameters: {
			candidate_labels: labels,
		},
	});

	return response;
}

async function extractFilterCriteria(query) {
	const criteria = {
		location: null,
		jobTitle: null,
		jobType: null,
		company: null,
	};

	const labels = ["location", "job title", "company", "job type"];
	const words = query.split(" ");

	for (const word of words) {
		const result = await classifyText(word, labels);

		let highestScoreLabel = null;
		let score = 0;

		if (Array.isArray(result) && result.length > 0) {
			highestScoreLabel = result[0]?.label ?? null;
			score = result[0]?.score ?? 0;
		} else if (result?.labels?.length) {
			highestScoreLabel = result.labels[0];
			score = result.scores?.[0] ?? 0;
		}

		if (score > 0.5) {
			switch (highestScoreLabel) {
				case "location":
					criteria.location = word;
					break;
				case "job title":
					criteria.jobTitle = word;
					break;
				case "company":
					criteria.company = word;
					break;
				case "job type":
					criteria.jobType = word;
					break;
				default:
					break;
			}
		}
	}

	return criteria;
}

async function performSimilaritySearch(collection, queryTerm, jobPostings) {
	try {
		const queryEmbedding = await generateEmbeddings([queryTerm]);
		const results = await queryCollection(collection.id, {
			query_embeddings: queryEmbedding,
			n_results: 3,
			include: ["distances", "documents", "metadatas"],
		});

		if (!results || results.length === 0 || !results?.ids?.[0]?.length) {
			console.log(`No similar results found for "${queryTerm}".`);
			return [];
		}

		const topJobPostings = results.ids[0]
			.map((id, index) => {
				const jobPosting = jobPostings.find((item) => item.jobId.toString() === id);
				if (!jobPosting) {
					return null;
				}

				return {
					id,
					jobTitle: jobPosting.jobTitle,
					jobType: jobPosting.jobType,
					jobDescription: jobPosting.jobDescription,
					company: jobPosting.company,
					score: results.distances?.[0]?.[index] ?? Number.MAX_SAFE_INTEGER,
				};
			})
			.filter(Boolean)
			.sort((a, b) => a.score - b.score);

		return topJobPostings;
	} catch (error) {
		console.error("Error during similarity search:", error.message);
		return [];
	}
}

main();
