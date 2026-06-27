import "dotenv/config";
const hfToken = process.env.HF_TOKEN;
if (!hfToken) {
    throw new Error("Missing HF_TOKEN environment variable.");
}
import { HfInference } from "@huggingface/inference";
const hf = new HfInference(hfToken);
import fs from "node:fs";
import { PDFParse } from "pdf-parse";

const filePath = "foodMenu.pdf";

const extractTextFromPDF = async (filePath) => {
  let parser;
  try {
    const dataBuffer = fs.readFileSync(filePath);
    parser = new PDFParse({ data: dataBuffer });
    const data = await parser.getText();
    const text = data.text.replace(/\n/g, " ").replace(/ +/g, " ");
    return text;
  } catch (err) {
    console.error("Error extracting text from PDF:", err);
    throw err;
  } finally {
    if (parser) {
      await parser.destroy();
    }
  }
};

const convertTextToEmbedding = async (text) => {
  try {
    const result = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: text,
    });
    // console.log("Embedding Result:", result);
    return result; // Return the embedding array
  } catch (err) {
    console.error("Error converting text to embeddings:", err);
    throw err;
  }
 };


async function main(){
  const text = await extractTextFromPDF(filePath);
  console.log("Extracted Text:", text);
  const embeddings=await convertTextToEmbedding(text);
  console.log(embeddings);
}

main();