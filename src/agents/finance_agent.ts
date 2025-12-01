import type { ChatOpenAI } from "@langchain/openai";
import type { RetrieverLike } from "./domain_agent.js";
import { DomainRagAgent } from "./domain_agent.js";

export async function createFinanceAgent(llm: ChatOpenAI, retriever: RetrieverLike) {
  return DomainRagAgent.init({
    llm,
    retriever,
    options: {
      intent: "finance",
      name: "Finance Operations Advisor",
      styleGuide:
        "Detail approval matrices, cite invoice and audit codes, include timelines, and flag SOX or budget compliance considerations explicitly."
    }
  });
}
