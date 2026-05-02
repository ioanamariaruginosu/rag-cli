# ask-ai-pdf

A small command-line RAG tool. Point it at a folder of PDFs and ask questions about them in plain English.

## Install

```bash
npm install -g ask-ai-pdf
```

## Setup

You need an OpenAI API key. Get one at https://platform.openai.com/api-keys.

The first time you run `ask-ai`, if no key is found it will prompt you to paste one and save it to `~/.config/ask-ai-pdf/config.json` (mode `0600`) so you only enter it once.

If you'd rather use an environment variable, set it instead and the prompt is skipped:

```bash
export OPENAI_API_KEY="sk-..."
```

(Add that line to `~/.zshrc` or `~/.bashrc` to make it permanent.)

To change the saved key later, just delete `~/.config/ask-ai-pdf/config.json` and you'll be prompted again.

## Use

`cd` into any folder containing PDFs, then run:

```bash
ask-ai
```

Or to load a single specific PDF:

```bash
ask-ai my-document.pdf
```

You'll get an interactive prompt:

```
📂 Loaded 12 PDF(s).
🧠 Database ready.
📝 Summaries ready.

💬 RAG CLI Active. Type your question (or 'exit' to quit):

You: what does the contract say about cancellation?
AI: ...
```

Type `exit` to quit.

## How it works

- Loads every PDF in the current directory (or the one you named).
- Splits each into 1000-character chunks and embeds them into an in-memory vector store using OpenAI embeddings.
- On every question, retrieves the 6 most relevant chunks and asks `gpt-4o` to answer.
- Also pre-computes a one-line summary per PDF (with the cheaper `gpt-4o-mini`) so it can answer corpus-level questions like "what files did you load?" or "summarize each PDF".

## Cost

Roughly:
- A one-time embedding cost per folder (proportional to total PDF text).
- ~$0.001 per PDF for the startup summary.
- ~$0.01–0.05 per question for the answer (depends on excerpt size).

The vector index is in-memory only, so re-running re-embeds everything.

## Requirements

- Node.js 20 or newer.
- An OpenAI API key.

## License

MIT
