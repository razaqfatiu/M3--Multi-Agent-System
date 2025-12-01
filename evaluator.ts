import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StructuredOutputParser } from "langchain/output_parsers";
import type { BaseMessage } from "@langchain/core/messages";
import { z } from "zod";
import { Langfuse } from "langfuse";
import { fileURLToPath } from "url";

const schema = z.object({
  score: z.number().min(1).max(10),
  reasoning: z.string().min(10),
});

const instructionsParser = StructuredOutputParser.fromZodSchema(schema);
const formatInstructions = instructionsParser.getFormatInstructions();

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are the QA evaluator. Score helpfulness on a 1-10 scale considering relevance, accuracy, and completeness."
  ],
  [
    "human",
    "Question: {question}\nAnswer: {answer}\n\nProvide JSON following: {format_instructions}"
  ]
]);

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? "";
const OPENROUTER_BASE_URL =
  process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1";
const EVALUATOR_MODEL =
  process.env.EVALUATOR_OPENROUTER_MODEL ??
  process.env.OPENROUTER_EVALUATOR_MODEL ??
  "openai/gpt-5-mini";

const llm = new ChatOpenAI({
  apiKey: OPENROUTER_API_KEY,
  model: EVALUATOR_MODEL,
  temperature: 0,
  configuration: {
    baseURL: OPENROUTER_BASE_URL,
  },
});

const langfuseClient = process.env.LANGFUSE_SECRET_KEY
  ? new Langfuse({
      publicKey: process.env.LANGFUSE_PUBLIC_KEY,
      secretKey: process.env.LANGFUSE_SECRET_KEY,
      baseUrl: process.env.LANGFUSE_BASE_URL ?? "https://cloud.langfuse.com",
    })
  : undefined;

export interface EvaluationResult {
  score: number;
  reasoning: string;
}

export async function evaluateAnswer(
  question: string,
  answer: string,
  traceId?: string
): Promise<EvaluationResult> {
  const messages = await prompt.formatMessages({
    question,
    answer,
    format_instructions: instructionsParser.getFormatInstructions()
  });
  const response = await llm.invoke(messages);
  const parsed = await instructionsParser.parse(extractText(response));
  const result: EvaluationResult = {
    score: parsed.score,
    reasoning: parsed.reasoning,
  };

  if (langfuseClient && traceId) {
    await langfuseClient.score({
      traceId,
      name: "rag-quality",
      value: result.score,
      comment: result.reasoning,
    });
  }

  return result;
}

function extractText(message: BaseMessage): string {
  const content = message.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((chunk) => {
        if (typeof chunk === "string") {
          return chunk;
        }
        if ("text" in chunk && typeof chunk.text === "string") {
          return chunk.text;
        }
        return "";
      })
      .join("")
      .trim();
  }
  return "";
}

const evaluatorFilePath = fileURLToPath(import.meta.url);

if (process.argv[1] === evaluatorFilePath) {
  const question = process.argv[2] ?? "How do I expense a work trip?";
  const answer = process.argv[3] ?? "Submit receipts within 10 days via Coupa.";
  evaluateAnswer(question, answer)
    .then((res) => console.log("Evaluator score", res))
    .catch((err) => {
      console.error("Evaluation failed", err);
      process.exitCode = 1;
    });
}
