import { HfInference } from "@huggingface/inference";

const hfToken = process.env.HF_TOKEN;
if (!hfToken) {
    throw new Error("Missing HF_TOKEN environment variable.");
}
const fh = new HfInference(hfToken);

const text = "Let's use a hugging face AI model";

const getEmbeddings = async () => {
    try {
        const embeddings = await convertTextToEmbedding(text);
        console.log(embeddings);
    } catch (err) {
        console.error("Error getting embeddings: ", err);
    }
};

const convertTextToEmbedding = async (text) => { 
    const response = await fh.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: text,
    });
    return response;
};

getEmbeddings();