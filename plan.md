# GemmaAgent Build Plan

## Goal

Build a fully local AI agent with RAG and tool use, running on an M1 Max MacBook Pro with 64GB RAM. Gemma 4 is installed locally via LM Studio. The architecture (RAG, tool use) is already understood — this plan is about implementation.

---

## Step 1 — Verify LM Studio API

Write a minimal Python script that sends a chat completion request to LM Studio's OpenAI-compatible endpoint at `localhost:1234/v1`. This confirms your local LLM is callable from code before adding any complexity.

---

## Step 2 — Build the RAG Pipeline (Ingest Side)

Pick a document loader strategy for mixed file types, chunk your documents, embed them with a local embedding model (sentence-transformers with `all-MiniLM-L6-v2` — tiny and fast on M1), and store vectors in ChromaDB. You'll have a CLI script that points at a folder and indexes everything.

---

## Step 3 — Build the RAG Pipeline (Retrieval Side)

Write a retrieval function that takes a query, embeds it, searches ChromaDB, and returns the top-k chunks with metadata. Test it standalone before wiring it into the agent.

---

## Step 4 — Implement Tool Definitions

Define your tools (web search, file ops, shell, custom API, and RAG retrieval itself) as plain Python functions, each with a schema describing its name, parameters, and purpose. This schema is what the LLM will see.

---

## Step 5 — Build the ReAct Agent Loop

This is the orchestration core. Send the user query + tool schemas to Gemma 4 via the OpenAI function-calling API. Parse the model's response — if it calls a tool, execute it and feed the result back. Loop until it produces a final answer or hits a max iteration cap.

---

## Step 6 — Wire It All Together into a REPL

A simple interactive loop where you type queries, the agent reasons, calls tools, retrieves knowledge, and responds.
