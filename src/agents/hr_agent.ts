import type { ChatOpenAI } from "@langchain/openai";
import type { RetrieverLike } from "./domain_agent.js";
import { DomainRagAgent } from "./domain_agent.js";

export async function createHrAgent(llm: ChatOpenAI, retriever: RetrieverLike) {
  return DomainRagAgent.init({
    llm,
    retriever,
    options: {
      intent: "hr",
      name: "HR Knowledge Specialist",
      styleGuide:
        "Prioritize empathy, cite policy IDs, mention leave types, onboarding steps, and benefits clarifications. Provide action items and escalation options for HRBP involvement."
    }
  });
}
