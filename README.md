# Multi-Agent Department Router

This project implements a LangChain-driven multi-agent system that classifies employee questions and routes them to specialized RAG agents (HR, IT/Tech, Finance). Each agent grounds answers in domain-specific documentation, while Langfuse tracing captures every step for observability. A bonus evaluator agent scores each response before it reaches an end user.

## Repository Structure
- `src/multi_agent_system.ts` – Main orchestration entry point organized into setup, document loading, agent wiring, router logic, demo harness, and Langfuse integration sections.
- `src/agents/` – Specialized agents (`hr_agent.ts`, `tech_agent.ts`, `finance_agent.ts`), orchestrator classifier, shared domain agent helper, and shared types.
- `data/<domain>_docs/` – Domain document collections (≥60 sections each) that ensure at least 50 retrievable chunks per department.
- `test_queries.json` – Ten intent-labeled prompts for regression checks and routing validation.
- `evaluator.ts` – Bonus evaluator agent that uses LangChain + Langfuse scores API to grade answers (1-10 scale).
- `.env.example` – Environment template for OpenRouter + Langfuse keys.

## Setup Instructions
1. **Install dependencies**
   ```bash
   npm install
   ```
2. **Configure environment** – copy `.env.example` to `.env` and set:
   - `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` (defaults to `https://openrouter.ai/api/v1`).
   - `OPENROUTER_MODEL` (router + agents) and `OPENROUTER_EMBEDDING_MODEL`.
   - `OPENROUTER_EMBEDDING_DIM` (defaults to `1024`) – set this to your Pinecone index dimension. The script enforces the Pinecone free-tier ceiling (1536), so higher values fall back to 1024 automatically.
   - `EVALUATOR_OPENROUTER_MODEL` (optional) – override the QA scorer model so the evaluator runs on a different OpenRouter model than the agents.
   - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`/`LANGFUSE_HOST` for tracing + scoring.
   - `PINECONE_API_KEY`, `PINECONE_INDEX`, and optional `PINECONE_CONTROLLER_HOST` so the router can seed/query Pinecone. `PINECONE_NAMESPACE_PREFIX` controls namespace names (default `dept`) and `PINECONE_SKIP_SEED=true` skips re-uploading docs on each run (useful once the corpora are already in the index).
   - Ensure your Pinecone index dimension matches the selected embedding model (e.g., `text-embedding-3-large` = 3072 dimensions).

## Running the Multi-Agent System
Execute the orchestrator (loads docs, builds vector stores, and runs sample questions):
```bash
npm start
```
Environment variables control which OpenRouter model is used, so you can swap providers without code changes. Langfuse tracing automatically wraps orchestrator + agent calls when keys are present; the CLI output shows query intent, reasoning, answers, and cited sources.

> **Notebook note:** This repo runs entirely via the `src/multi_agent_system.ts` entrypoint instead of a notebook. If you prefer a notebook workflow, run the same sequence: install deps → load env vars → execute the TypeScript entrypoint via `ts-node` or `npm start`.

### Running a Single Query via CLI
Pass the question after `npm start` and the router will only evaluate that prompt:
```bash
npm start -- "How do I request emergency PTO and restore VPN access?"
```
The CLI run is traced with `query_type: "cli"` and prints every agent turn (answers, sources, handoff info). Without extra arguments, the script falls back to the bundled sample queries.

### How Routing & Handoffs Work
- **Multi-intent classification** – the orchestrator (LangChain prompt + Zod schema) always returns an ordered list of departments. If a question mixes topics (e.g., HR + Tech), every relevant agent is queued sequentially.
- **Context packages during handoff** – each domain agent emits structured JSON that includes the written answer, citations, and (optionally) a `follow_up` block containing the next intent, rationale, and a short context brief. The router forwards that note so the next agent sees the running transcript plus the specific follow-up directive.
- **Out-of-scope detection** – when no confident intent is found, `unknown` remains in the classification array and the CLI prints that the request is outside supported departments instead of guessing.
- **Extending the router** – import `MultiAgentRouter` from `src/multi_agent_system.ts` to embed this workflow in another service. The router exposes every agent “turn” (intent, answer, sources, handoff signal) so downstream systems can display or audit the entire conversation.

## Evaluator Agent (Bonus)
The evaluator scores answers using the same OpenRouter model and pushes qualitative metrics to Langfuse:
```bash
npx ts-node evaluator.ts "How do I expense a work trip?" "Submit receipts within 10 days via Coupa."
```
Pass a Langfuse trace ID as the third argument when embedding into workflows so the score attaches to the correct trace via `evaluateAnswer(question, answer, traceId)`.

## Test Queries & Regression
Use `test_queries.json` to validate routing quickly. Example harness snippet:
```ts
import queries from "./test_queries.json" assert { type: "json" };
for (const { query, expected_intent } of queries) {
  const result = await router.route(query, traceConfig);
  console.log(query, result.routedIntent, expected_intent);
}
```
The dataset covers each department plus an `unknown` edge case.

## Technical Decisions
- **LangChain everywhere** – Chat models, retrievers, and Runnables keep the architecture composable and observable, instead of custom prompts wired by hand.
- **DirectoryLoader + RecursiveCharacterTextSplitter + Pinecone** – ensures ≥50 granular chunks per department for precise retrieval without over-fetching tokens, while Pinecone namespaces preserve embeddings between runs.
- **Domain agents with explicit JSON contracts** – each department-specific Retrieval-Augmented agent returns `{ answer, citations, follow_up }`, allowing the orchestrator to reason about handoffs programmatically instead of parsing prose.
- **Multi-intent orchestrator** – Zod + `StructuredOutputParser` enforce an ordered array of intents, enabling sequential delegation (HR → Tech → Finance) when a request spans multiple functions.
- **Langfuse tracing** – `LangfuseCallbackHandler` lets us trace orchestrator + agent chains, while `evaluator.ts` posts feedback scores to the same trace for QA dashboards.
- **OpenRouter adapter** – all models/embeddings take `OPENROUTER_BASE_URL`, so swapping Anthropic/OpenAI/etc. happens entirely via env configuration.
- **Pinecone vector store** – the router seeds each department into isolated Pinecone namespaces (configurable prefix) using deterministic chunk IDs so embeddings persist between runs. Set `PINECONE_SKIP_SEED=true` if you want to reuse the existing namespace without clearing it.

## Known Limitations & Next Steps
- Default Pinecone seeding wipes and re-ingests departmental namespaces on each run; flip `PINECONE_SKIP_SEED=true` or add a smarter ETL step before moving to production data.
- Evaluator currently uses the same model as the agents; adopting a specialized judge model may yield better metrics.
- Authentication, rate limiting, and streaming responses are out of scope but straightforward to add through LangChain’s router APIs.
