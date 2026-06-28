import "dotenv/config";
import fs from "node:fs";
import readline from "node:readline";
import { PDFParse } from "pdf-parse";
import { HfInference } from "@huggingface/inference";
import { ChromaClient } from "chromadb";
import jobPostings from "./jobPostings.js";

const hfToken = process.env.HF_TOKEN;
if (!hfToken) {
	throw new Error("Missing HF_TOKEN environment variable.");
}

const hf = new HfInference(hfToken);
const client = new ChromaClient({ path: "http://localhost:8000" });
const collectionName = "smart_job_collection";
const chromaBaseUrl = "http://localhost:8000/api/v2";
const chromaTenant = "default_tenant";
const chromaDatabase = "default_database";

const createReadlineInterface = () =>
	readline.createInterface({
		input: process.stdin,
		output: process.stdout,
	});

function promptUserInput() {
	const rl = createReadlineInterface();

	return new Promise((resolve) => {
		rl.question("Enter the full path and file name of the PDF resume: ", (answer) => {
			rl.close();
			resolve(answer.trim());
		});
	});
}

async function extractFromPDF(filePath) {
	let parser;
	try {
		const dataBuffer = await fs.promises.readFile(filePath);
		parser = new PDFParse({ data: dataBuffer });
		const data = await parser.getText();
		const text = data.text.replace(/\s+/g, " ").trim();

		return { dataBuffer, text };
	} catch (error) {
		console.error("Error extracting data from PDF:", error);
		throw error;
	} finally {
		if (parser) {
			await parser.destroy();
		}
	}
}

async function extractTextFromPDF(filePath) {
	let parser;
	try {
		const dataBuffer = fs.readFileSync(filePath);
		parser = new PDFParse({ data: dataBuffer });
		const data = await parser.getText();
		return data.text.replace(/\s+/g, " ").trim();
	} catch (error) {
		console.error("Error extracting text from PDF:", error);
		throw error;
	} finally {
		if (parser) {
			await parser.destroy();
		}
	}
}

async function generateEmbeddings(text) {
	const results = await hf.featureExtraction({
		model: "sentence-transformers/all-MiniLM-L6-v2",
		inputs: text,
	});

	return results;
}

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

async function storeEmbeddings() {
	try {
		const jobEmbeddings = [];
		const metadatas = jobPostings.map(() => ({}));
		const ids = jobPostings.map((_, index) => `job_${index}`);
		const documents = jobPostings.map((jobPosting) => jobPosting.jobTitle.toLowerCase());

		for (const [index, jobPosting] of jobPostings.entries()) {
			const jobText = [
				jobPosting.jobTitle,
				jobPosting.jobDescription,
				jobPosting.jobType,
				jobPosting.location,
			]
				.join(" ")
				.toLowerCase();

			const embedding = await generateEmbeddings(jobText);
			jobEmbeddings[index] = embedding;
		}

		const collection = await getOrCreateCollection(collectionName);
		await upsertCollectionRecords(collection.id, {
			ids,
			documents,
			embeddings: jobEmbeddings,
			metadatas,
		});

		return collection;
	} catch (error) {
		console.error("Error storing embeddings:", error);
		throw error;
	}
}

async function storeEmbedding() {
	await storeEmbeddings();
}

async function main() {
	try {
		await storeEmbedding();

		const filePath = await promptUserInput();
		const text = await extractTextFromPDF(filePath);
		const resumeEmbedding = await generateEmbeddings(text);

		const collection = await getOrCreateCollection(collectionName);
		const results = await queryCollection(collection.id, {
			query_embeddings: [resumeEmbedding],
			n_results: 5,
			include: ["distances", "documents", "metadatas"],
		});

		if (results?.ids?.[0]?.length) {
			results.ids[0].forEach((id) => {
				const jobIndex = Number.parseInt(id.replace("job_", ""), 10);
				const recommendedJob = jobPostings[jobIndex];
				if (recommendedJob) {
					console.log(`Recommended Job: ${recommendedJob.jobTitle}`);
				}
			});
		} else {
			console.log("No similar job recommendations found.");
		}
	} catch (error) {
		console.error("Error in main function:", error);
		throw error;
	}
}

console.log("Environment initialized for smart recommendation system.");
console.log(`Loaded job postings: ${jobPostings.length}`);

void client;
void extractFromPDF;

main();
