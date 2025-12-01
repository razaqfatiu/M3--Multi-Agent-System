import type { ChatOpenAI } from "@langchain/openai";
import type { RetrieverLike } from "./domain_agent.js";
import { DomainRagAgent } from "./domain_agent.js";

export async function createTechAgent(llm: ChatOpenAI, retriever: RetrieverLike) {
  return DomainRagAgent.init({
    llm,
    retriever,
    options: {
      intent: "tech",
      name: "IT Support Strategist",
      styleGuide:
        "Diagnose root causes, reference KB tickets, surface remediation steps with command examples, and list monitoring signals before resolving incidents."
    }
  });
}
