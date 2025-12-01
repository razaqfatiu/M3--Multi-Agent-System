export type DepartmentIntent = "hr" | "tech" | "finance" | "unknown";

export interface AgentAnswer {
  intent: DepartmentIntent;
  answer: string;
  sources: string[];
  confidence: number;
  rationale?: string;
}

export interface AgentContext {
  langfuseTraceId?: string;
}
