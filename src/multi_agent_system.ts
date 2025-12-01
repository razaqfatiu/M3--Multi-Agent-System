/**
 * ## 1. Setup & Imports
 */
import 'dotenv/config';
import path from 'path';
import { fileURLToPath } from 'url';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CSVLoader } from '@langchain/community/document_loaders/fs/csv';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import type { DocumentInterface } from '@langchain/core/documents';
import type { RunnableConfig } from '@langchain/core/runnables';
import { CallbackHandler as LangfuseCallbackHandler } from '@langfuse/langchain';
import { Pinecone } from '@pinecone-database/pinecone';
import type { Index } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { OrchestratorAgent } from './agents/orchestrator.js';
import { createHrAgent } from './agents/hr_agent.js';
import { createTechAgent } from './agents/tech_agent.js';
import { createFinanceAgent } from './agents/finance_agent.js';
import type { DepartmentIntent } from './agents/types.js';
import { DomainRagAgent } from './agents/domain_agent.js';
import type {
  RetrieverLike,
  DomainAgentResult,
} from './agents/domain_agent.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const dataDir = path.join(rootDir, 'data');

const OPENROUTER_BASE_URL =
  process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? '';
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL ?? 'gpt-4o-mini';
const EMBEDDING_MODEL =
  process.env.OPENROUTER_EMBEDDING_MODEL ?? 'text-embedding-3-large';
const EMBEDDING_DIM = (() => {
  const raw = process.env.OPENROUTER_EMBEDDING_DIM ?? '1024';
  const parsed = Number(raw);
  if (Number.isNaN(parsed)) {
    throw new Error('OPENROUTER_EMBEDDING_DIM must be a valid number.');
  }
  if (parsed > 1536) {
    console.warn(
      'OPENROUTER_EMBEDDING_DIM exceeds Pinecone free plan max (1536). Using 1024.'
    );
    return 1024;
  }
  return parsed;
})();

const PINECONE_API_KEY = process.env.PINECONE_API_KEY ?? '';
const PINECONE_INDEX = process.env.PINECONE_INDEX ?? '';
const PINECONE_CONTROLLER_HOST = process.env.PINECONE_CONTROLLER_HOST;
const PINECONE_NAMESPACE_PREFIX =
  process.env.PINECONE_NAMESPACE_PREFIX ?? 'dept';
const SHOULD_SEED_PINECONE = process.env.PINECONE_SKIP_SEED !== 'true';

/**
 * ## 2. Document Loading & Vector Stores
 */
async function loadDocuments(
  domainFolder: string
): Promise<DocumentInterface[]> {
  const folderPath = path.join(dataDir, domainFolder);
  const loader = new DirectoryLoader(folderPath, {
    '.md': (filePath: string) => new TextLoader(filePath),
    '.txt': (filePath: string) => new TextLoader(filePath),
    '.csv': (filePath: string) => new CSVLoader(filePath),
  });
  const docs = await loader.load();
  return docs;
}

let pineconeIndexPromise: Promise<Index> | null = null;

async function resolvePineconeIndex(): Promise<Index> {
  if (!PINECONE_API_KEY || !PINECONE_INDEX) {
    throw new Error(
      'Pinecone configuration missing. Set PINECONE_API_KEY and PINECONE_INDEX.'
    );
  }
  if (!pineconeIndexPromise) {
    const pinecone = new Pinecone({
      apiKey: PINECONE_API_KEY,
      ...(PINECONE_CONTROLLER_HOST
        ? { controllerHostUrl: PINECONE_CONTROLLER_HOST }
        : {}),
    });
    pineconeIndexPromise = Promise.resolve(pinecone.index(PINECONE_INDEX));
  }
  return pineconeIndexPromise;
}

async function buildVectorStore(
  domainFolder: string,
  embeddings: OpenAIEmbeddings,
  pineconeIndex: Index
): Promise<PineconeStore> {
  const docs = await loadDocuments(domainFolder);
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 40,
  });
  const splitDocs = await splitter.splitDocuments(docs);
  const namespace = namespaceForDomain(domainFolder);
  const store = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
    namespace,
  });
  if (SHOULD_SEED_PINECONE) {
    try {
      await store.delete({ deleteAll: true, namespace });
    } catch (error: any) {
      if (typeof error?.message === 'string' && error.message.includes('404')) {
        console.warn(
          `Pinecone namespace ${namespace} not found yet. Skipping delete.`
        );
      } else {
        throw error;
      }
    }
    const ids = splitDocs.map((_, idx) => `${namespace}-${idx}`);
    await store.addDocuments(splitDocs, { ids, namespace });
  }
  return store;
}

function createRetriever(store: PineconeStore): RetrieverLike {
  return store.asRetriever({ k: 5 }) as unknown as RetrieverLike;
}

/**
 * ## 3. Agent Definitions
 */
async function buildAgents(
  llm: ChatOpenAI,
  stores: Record<string, PineconeStore>
) {
  const hrAgent = await createHrAgent(llm, createRetriever(stores.hr));
  const techAgent = await createTechAgent(llm, createRetriever(stores.tech));
  const financeAgent = await createFinanceAgent(
    llm,
    createRetriever(stores.finance)
  );

  const agentMap = {
    hr: hrAgent,
    tech: techAgent,
    finance: financeAgent,
  } satisfies Record<Exclude<DepartmentIntent, 'unknown'>, DomainRagAgent>;

  return agentMap;
}

function namespaceForDomain(domainFolder: string): string {
  const slug = domainFolder
    .replace(/_docs?$/i, '')
    .replace(/[^a-z0-9]+/gi, '-');
  return `${PINECONE_NAMESPACE_PREFIX}-${slug}`.toLowerCase();
}

/**
 * ## 4. Orchestrator & Routing
 */
interface AgentTurn {
  intentTried: DepartmentIntent;
  response: DomainAgentResult;
}

interface IntentQueueItem {
  intent: DepartmentIntent;
  note?: string;
}

interface RouteResult {
  classification: Awaited<ReturnType<OrchestratorAgent['classify']>>;
  turns: AgentTurn[];
  unresolvedIntents: DepartmentIntent[];
}

export class MultiAgentRouter {
  constructor(
    private readonly orchestrator: OrchestratorAgent,
    private readonly agents: Record<
      Exclude<DepartmentIntent, 'unknown'>,
      DomainRagAgent
    >
  ) {}

  async route(question: string, config?: RunnableConfig): Promise<RouteResult> {
    const classification = await this.orchestrator.classify(question, config);
    const orderedIntents =
      OrchestratorAgent.resolveOrderedIntents(classification);
    const queue: IntentQueueItem[] = orderedIntents.map((intent) => ({
      intent,
    }));
    const visited = new Set<DepartmentIntent>();
    const turns: AgentTurn[] = [];

    while (queue.length && turns.length < 5) {
      const { intent, note } = queue.shift()!;
      if (intent === 'unknown' || visited.has(intent)) {
        continue;
      }
      const agent = this.agents[intent as Exclude<DepartmentIntent, 'unknown'>];
      if (!agent) {
        continue;
      }
      const history = turns
        .map((turn) => {
          const label =
            turn.intentTried === 'unknown'
              ? 'Unknown intent'
              : this.agents[turn.intentTried]?.name ?? turn.intentTried;
          return `${label}:\n${turn.response.text}`;
        })
        .join('\n\n');
      const taskDirective = note
        ? `${question}\n\nFollow-up directive: ${note}`
        : question;
      const response = await agent.invoke(
        taskDirective,
        history || 'No prior agent responses.',
        config
      );
      turns.push({ intentTried: intent, response });
      visited.add(intent);
      if (response.handoff && !visited.has(response.handoff.intent)) {
        queue.push({
          intent: response.handoff.intent,
          note: response.handoff.context ?? response.handoff.reason,
        });
      }
    }

    const unresolvedIntents = queue
      .map((item) => item.intent)
      .filter((intent) => !visited.has(intent));

    return {
      classification,
      turns,
      unresolvedIntents,
    };
  }
}

/**
 * ## 5. Testing & Examples
 */
async function runExamples(
  router: MultiAgentRouter,
  handler?: LangfuseCallbackHandler
) {
  const sampleQueries = [
    'How do I request paid family leave?',
    'VPN keeps timing out when I try to load Salesforce dashboards.',
    'What approvals do I need for a vendor over $50k?',
  ];

  for (const query of sampleQueries) {
    const traceConfig: RunnableConfig | undefined = handler
      ? { callbacks: [handler], metadata: { query_type: 'sample' } }
      : undefined;
    const result = await router.route(query, traceConfig);
    console.log('\n---');
    console.log('Query:', query);
    console.log('Ordered intents:', result.classification.intents.join(' → '));
    console.log('Confidence:', result.classification.confidence.toFixed(2));
    console.log('Reasoning:', result.classification.reasoning);
    if (!result.turns.length) {
      console.log('No specialized agent available for this request.');
      continue;
    }
    for (const turn of result.turns) {
      console.log(`\n[${turn.intentTried.toUpperCase()} AGENT]`);
      console.log(turn.response.text);
      console.log('Sources:', turn.response.sources.join(', '));
      if (turn.response.handoff) {
        console.log(
          `Handoff requested → ${turn.response.handoff.intent} (${turn.response.handoff.reason})`
        );
      }
    }
    if (result.unresolvedIntents.length) {
      console.log('Unresolved intents:', result.unresolvedIntents.join(', '));
    }
  }
}

/**
 * ## 6. Langfuse Integration
 */
async function configureLangfuse() {
  if (!process.env.LANGFUSE_SECRET_KEY || !process.env.LANGFUSE_PUBLIC_KEY) {
    console.warn('Langfuse keys missing. Tracing disabled.');
    return undefined;
  }
  return new LangfuseCallbackHandler({
    tags: ['multi-agent-router'],
    traceMetadata: {
      service: 'department-router',
      environment: process.env.NODE_ENV ?? 'local',
    },
  });
}

async function bootstrap() {
  const langfuseHandler = await configureLangfuse();
  const llm = new ChatOpenAI({
    temperature: 0,
    model: OPENROUTER_MODEL,
    apiKey: OPENROUTER_API_KEY,
    configuration: {
      baseURL: OPENROUTER_BASE_URL,
    },
  });

  const embeddings = new OpenAIEmbeddings({
    apiKey: OPENROUTER_API_KEY,
    model: EMBEDDING_MODEL,
    ...(EMBEDDING_DIM ? { dimensions: EMBEDDING_DIM } : {}),
    configuration: {
      baseURL: OPENROUTER_BASE_URL,
    },
  });

  const pineconeIndex = await resolvePineconeIndex();
  const stores = {
    hr: await buildVectorStore('hr_docs', embeddings, pineconeIndex),
    tech: await buildVectorStore('tech_docs', embeddings, pineconeIndex),
    finance: await buildVectorStore('finance_docs', embeddings, pineconeIndex),
  };

  const agents = await buildAgents(llm, stores);
  const orchestrator = new OrchestratorAgent(llm);
  const router = new MultiAgentRouter(orchestrator, agents);

  await runExamples(router, langfuseHandler);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  bootstrap().catch((error) => {
    console.error('Failed to bootstrap multi-agent system', error);
    process.exitCode = 1;
  });
}
