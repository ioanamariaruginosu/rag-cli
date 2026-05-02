#!/usr/bin/env node
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as readline from "readline/promises";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";

async function ensureApiKey() {
    if (process.env.OPENAI_API_KEY) return;

    const configDir = path.join(os.homedir(), ".config", "ask-ai-pdf");
    const configPath = path.join(configDir, "config.json");

    if (fs.existsSync(configPath)) {
        try {
            const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
            if (config.openaiApiKey) {
                process.env.OPENAI_API_KEY = config.openaiApiKey;
                return;
            }
        } catch {
            // fall through and re-prompt
        }
    }

    console.log("\n🔑 No OpenAI API key found.");
    console.log("   Get one at: https://platform.openai.com/api-keys");

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const key = (await rl.question("   Paste your key (saved to ~/.config/ask-ai-pdf/config.json): ")).trim();
    rl.close();

    if (!key) {
        console.error("❌ No key provided. Exiting.");
        process.exit(1);
    }

    fs.mkdirSync(configDir, { recursive: true });
    fs.writeFileSync(configPath, JSON.stringify({ openaiApiKey: key }, null, 2), { mode: 0o600 });
    process.env.OPENAI_API_KEY = key;
    console.log(`✅ Key saved.\n`);
}

function startSpinner(label: string) {
    const frames = [".  ", ".. ", "..."];
    let i = 0;
    process.stdout.write(`${label}${frames[0]}`);
    const interval = setInterval(() => {
        i = (i + 1) % frames.length;
        process.stdout.write(`\r${label}${frames[i]}`);
    }, 350);
    return () => {
        clearInterval(interval);
        process.stdout.write(`\r\x1b[K`);
    };
}

async function run() {
    await ensureApiKey();

    const currentFolder = process.cwd();
    const args = process.argv.slice(2);
    const specificFile = args[0];

    type LoadedFile = { filename: string; pages: any[] };
    const loadedFiles: LoadedFile[] = [];

    console.log(`\n🔍 Scanning files in: ${currentFolder}...`);

    try {
        const targets = specificFile
            ? [specificFile]
            : fs.readdirSync(currentFolder).filter(f => f.toLowerCase().endsWith(".pdf"));

        if (targets.length === 0) {
            throw new Error("No PDFs found in this directory.");
        }

        for (const file of targets) {
            const filePath = path.join(currentFolder, file);
            const loader = new PDFLoader(filePath);
            const pages = await loader.load();
            pages.forEach(p => { p.metadata.source = file; });
            loadedFiles.push({ filename: file, pages });
        }
        console.log(`📂 Loaded ${loadedFiles.length} PDF(s).`);
    } catch (error: any) {
        console.error(`❌ Error loading files: ${error.message}`);
        process.exit(1);
    }

    const docs = loadedFiles.flatMap(f => f.pages);

    const stopIndex = startSpinner("🧠 Building local database");
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const chunkedDocs = await textSplitter.splitDocuments(docs);
    const vectorStore = await MemoryVectorStore.fromDocuments(chunkedDocs, new OpenAIEmbeddings());
    stopIndex();
    console.log("🧠 Database ready.");

    const summaryLLM = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });
    const stopSum = startSpinner(`📝 Summarizing ${loadedFiles.length} PDF(s)`);
    const summaries = await Promise.all(loadedFiles.map(async f => {
        const text = f.pages.map(p => p.pageContent).join("\n").slice(0, 6000);
        const result = await summaryLLM.invoke(
            `Summarize what this PDF is about in 5-12 words. Reply with the summary only, no preamble or quotes.\n\nFilename: ${f.filename}\n\nContent:\n${text}`
        );
        return { filename: f.filename, summary: String(result.content).trim() };
    }));
    stopSum();
    console.log("📝 Summaries ready.");

    const fileList = summaries.map(s => `- ${s.filename}: ${s.summary}`).join("\n");
    const fileCount = loadedFiles.length;

    const llm = new ChatOpenAI({ modelName: "gpt-4o", temperature: 0 });
    const promptTemplate = PromptTemplate.fromTemplate(`You are an assistant answering questions about a collection of PDF documents the user has loaded.

You have access to {fileCount} loaded PDFs. Each is listed with a brief summary:
{fileList}

The following excerpts were retrieved as most relevant to the question. Each excerpt is labeled with its source filename:

{context}

Question: {input}

Instructions:
- For questions about which PDFs are loaded or what they contain at a high level, use the document list above — you DO have this information, never deny it.
- For specific factual questions, rely on the excerpts and cite source filenames.
- If neither the list nor the excerpts contain the answer, say so plainly.`);

    const retriever = vectorStore.asRetriever(6);

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    console.log("\n💬 RAG CLI Active. Type your question (or 'exit' to quit):");

    while (true) {
        const question = await rl.question("\nYou: ");
        if (question.toLowerCase() === "exit") break;

        const stop = startSpinner("AI is thinking");
        try {
            const retrieved = await retriever.invoke(question);
            const context = retrieved
                .map(d => `[Source: ${d.metadata.source ?? "unknown"}]\n${d.pageContent}`)
                .join("\n\n");
            const formatted = await promptTemplate.format({
                fileCount: String(fileCount),
                fileList,
                context,
                input: question,
            });
            const result = await llm.invoke(formatted);
            stop();
            console.log(`AI: ${String(result.content)}`);
        } catch (e: any) {
            stop();
            console.error(`AI: ⚠️ ${e.message}`);
        }
    }

    rl.close();
}

run();
