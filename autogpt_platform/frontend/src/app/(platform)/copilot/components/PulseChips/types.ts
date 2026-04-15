import type { AgentStatus } from "@/app/(platform)/library/types";

export interface PulseChipData {
  id: string;
  agentID: string;
  name: string;
  status: AgentStatus;
  shortMessage: string;
}
