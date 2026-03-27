#!/usr/bin/env tsx
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline/promises";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { PromptTemplate } from "@langchain/core/prompts";

async function run() {
    const currentFolder = process.cwd();
    const args = process.argv.slice(2);
    const specificFile = args[0];

    let docs: any[] = [];

    console.log(`\n🔍 Scanning files in: ${currentFolder}...`);

    try {
        if (specificFile) {
            const filePath = path.join(currentFolder, specificFile);
            const loader = new PDFLoader(filePath);
            docs = await loader.load();
            console.log(`📄 Loaded specific file: ${specificFile}`);
        } else {
            const allFiles = fs.readdirSync(currentFolder);
            const pdfFiles = allFiles.filter(file => file.toLowerCase().endsWith(".pdf"));

            if (pdfFiles.length === 0) {
                throw new Error("No PDFs found in this directory.");
            }

            for (const file of pdfFiles) {
                const filePath = path.join(currentFolder, file);
                const loader = new PDFLoader(filePath);
                const loadedDocs = await loader.load();
                docs.push(...loadedDocs);
            }
            console.log(`📂 Successfully loaded ${pdfFiles.length} PDF(s).`);
        }
    } catch (error: any) {
        console.error(`❌ Error loading files: ${error.message}`);
        process.exit(1);
    }

    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const chunkedDocs = await textSplitter.splitDocuments(docs);

    process.stdout.write("🧠 Thinking and building local database... ");
    const vectorStore = await MemoryVectorStore.fromDocuments(chunkedDocs, new OpenAIEmbeddings());
    console.log("Done!\n");

    const llm = new ChatOpenAI({ modelName: "gpt-4o", temperature: 0 });
    const prompt = PromptTemplate.fromTemplate(`
    Use the provided context to answer the question. If you don't know, just say you don't know.
    Context: {context}
    Question: {input}
  `);

    const combineDocsChain = await createStuffDocumentsChain({ llm, prompt });
    const retrievalChain = await createRetrievalChain({
        retriever: vectorStore.asRetriever(3),
        combineDocsChain,
    });

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    console.log("💬 RAG CLI Active. Type your question (or 'exit' to quit):");

    while (true) {
        const question = await rl.question("\nYou: ");
        if (question.toLowerCase() === "exit") break;

        process.stdout.write("AI: ");
        const response = await retrievalChain.invoke({ input: question });
        console.log(response.answer);
    }

    rl.close();
}

run();