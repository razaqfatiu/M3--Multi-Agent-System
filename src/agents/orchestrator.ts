import { ChatPromptTemplate } from "@langchain/core/prompts";
import type { ChatOpenAI } from "@langchain/openai";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseMessage } from "@langchain/core/messages";
import { DepartmentIntent } from "./types.js";

const schema = z.object({
  intents: z.array(z.enum(["hr", "tech", "finance", "unknown"] as const)).min(1).max(3),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().min(10)
});

export type OrchestratorResult = z.infer<typeof schema>;

const orchestratorParser = StructuredOutputParser.fromZodSchema(schema);

export class OrchestratorAgent {
  private readonly parser = orchestratorParser;
  private readonly prompt: ChatPromptTemplate;

  constructor(private readonly llm: ChatOpenAI) {
    const formatInstructions = this.parser.getFormatInstructions();
    this.prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are the routing orchestrator for the employee helpdesk. Identify whether a question spans multiple departments (hr, tech, finance, unknown) and order the intents in the sequence they should be engaged. Always mention unknown if nothing fits."
      ],
      [
        "human",
        "Question: {question}\n\nReturn JSON that follows: {format_instructions}"
      ]
    ]);
  }

  async classify(question: string, config?: RunnableConfig): Promise<OrchestratorResult> {
    const messages = await this.prompt.formatMessages({
      question,
      format_instructions: this.parser.getFormatInstructions()
    });
    const response = await this.llm.invoke(messages, config);
    return (await this.parser.parse(this.extractText(response))) as OrchestratorResult;
  }

  static resolveOrderedIntents(result: OrchestratorResult): DepartmentIntent[] {
    const unique: DepartmentIntent[] = [];
    for (const candidate of result.intents) {
      const normalized: DepartmentIntent =
        candidate === "hr" || candidate === "tech" || candidate === "finance" ? candidate : "unknown";
      if (!unique.includes(normalized)) {
        unique.push(normalized);
      }
    }
    return unique;
  }

  private extractText(message: BaseMessage): string {
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
}
