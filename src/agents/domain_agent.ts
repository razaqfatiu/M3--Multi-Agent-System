import type { RunnableConfig } from "@langchain/core/runnables";
import type { DocumentInterface } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import type { BaseMessage } from "@langchain/core/messages";
import type { ChatOpenAI } from "@langchain/openai";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";
import { DepartmentIntent } from "./types.js";

export interface RetrieverLike {
  invoke(input: string, config?: RunnableConfig): Promise<DocumentInterface[]>;
}

export interface DomainAgentOptions {
  intent: DepartmentIntent;
  name: string;
  styleGuide: string;
}

export interface DomainAgentInput {
  llm: ChatOpenAI;
  retriever: RetrieverLike;
  options: DomainAgentOptions;
}

export interface DomainAgentResult {
  text: string;
  sources: string[];
  handoff?: {
    intent: DepartmentIntent;
    reason: string;
    context?: string;
  };
}

export class DomainRagAgent {
  private readonly prompt: ChatPromptTemplate;
  private readonly parser = StructuredOutputParser.fromZodSchema(
    z.object({
      answer: z.string(),
      citations: z.array(z.string()),
      follow_up: z
        .object({
          intent: z.enum(["hr", "tech", "finance", "unknown"] as const),
          reason: z.string(),
          context_package: z.string().optional()
        })
        .optional()
    })
  );
  private readonly options: DomainAgentOptions;

  private constructor(
    private readonly llm: ChatOpenAI,
    private readonly retriever: RetrieverLike,
    options: DomainAgentOptions,
    prompt: ChatPromptTemplate
  ) {
    this.prompt = prompt;
    this.options = options;
  }

  static async init({ llm, retriever, options }: DomainAgentInput): Promise<DomainRagAgent> {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are ${options.name}. Follow this style guide: ${options.styleGuide}. ` +
          "Ground every reply in the provided context, cite KB identifiers when possible, and say you do not know if the answer is missing. If another department must help, clearly state that in the follow_up section of the JSON response described by {format_instructions}."
      ],
      [
        "human",
        "Conversation so far:\n{history}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAdhere to: {format_instructions}"
      ]
    ]);

    return new DomainRagAgent(llm, retriever, options, prompt);
  }

  async invoke(
    question: string,
    history = "No prior agent responses.",
    config?: RunnableConfig
  ): Promise<DomainAgentResult> {
    const sourceDocs = await this.retriever.invoke(question, config);
    const context = this.formatContext(sourceDocs);
    const messages = await this.prompt.formatMessages({
      context,
      question,
      history,
      format_instructions: this.parser.getFormatInstructions()
    });
    const llmResponse = await this.llm.invoke(messages, config);
    const parsed = await this.parser.parse(this.extractAnswer(llmResponse));
    return {
      text: parsed.answer,
      sources:
        parsed.citations.length > 0
          ? parsed.citations
          : sourceDocs.map((doc, idx) => {
              const source = doc.metadata?.source;
              return typeof source === "string" ? source : `chunk-${idx}`;
            }),
      handoff:
        parsed.follow_up && parsed.follow_up.intent !== "unknown"
          ? {
              intent: parsed.follow_up.intent,
              reason: parsed.follow_up.reason,
              context: parsed.follow_up.context_package
            }
          : undefined
    };
  }

  private formatContext(docs: DocumentInterface[]): string {
    if (!docs.length) {
      return "No matching documents.";
    }
    return docs
      .map((doc, idx) => {
        const sourceLabel = doc.metadata?.source ?? `chunk-${idx}`;
        return `Source: ${sourceLabel}\n${doc.pageContent}`;
      })
      .join("\n\n---\n\n");
  }

  private extractAnswer(message: BaseMessage): string {
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

  get intent(): DepartmentIntent {
    return this.options.intent;
  }

  get name(): string {
    return this.options.name;
  }
}
