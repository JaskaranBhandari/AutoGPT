import { act, cleanup, renderHook } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// --- Toast mock (must be stable across rerenders) ---
const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

// --- Environment mock ---
vi.mock("@/services/environment", () => ({
  environment: { getAGPTServerBaseUrl: () => "http://localhost:8006" },
}));

// --- API endpoints mock ---
const mockCancelSessionTask = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  postV2CancelSessionTask: (...args: unknown[]) =>
    mockCancelSessionTask(...args),
  getGetV2GetCopilotUsageQueryKey: () => ["usage"],
  getGetV2GetSessionQueryKey: (id: string) => ["session", id],
}));

// --- React Query mock ---
const mockInvalidateQueries = vi.fn();
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }),
}));

// --- Helpers mock ---
const mockHasActiveBackendStream = vi.fn(
  (_result: { data?: unknown }) => false,
);
const mockDisconnectSessionStream = vi.fn((_sid: string) => {});
vi.mock("../helpers", () => ({
  getCopilotAuthHeaders: vi.fn(async () => ({ Authorization: "Bearer test" })),
  deduplicateMessages: (msgs: UIMessage[]) => msgs,
  extractSendMessageText: (arg: unknown) =>
    arg && typeof arg === "object" && "text" in arg
      ? String((arg as { text: string }).text)
      : String(arg ?? ""),
  hasActiveBackendStream: (result: { data?: unknown }) =>
    mockHasActiveBackendStream(result),
  hasVisibleAssistantContent: (messages: UIMessage[]) => {
    const last = messages[messages.length - 1];
    if (last?.role !== "assistant") return false;
    return last.parts.some((part: UIMessage["parts"][number]) => {
      if (part.type === "text" && part.text.trim().length > 0) return true;
      if (part.type === "reasoning" && part.text.trim().length > 0) return true;
      if (part.type.startsWith("tool-")) return true;
      return false;
    });
  },
  resolveInProgressTools: (msgs: UIMessage[]) => msgs,
  getSendSuppressionReason: () => null,
  disconnectSessionStream: (sid: string) => mockDisconnectSessionStream(sid),
}));

// --- ai SDK mock (DefaultChatTransport must be constructible) ---
vi.mock("ai", () => ({
  DefaultChatTransport: vi.fn().mockImplementation(function () {
    return {};
  }),
}));

// --- @ai-sdk/react useChat mock with callback capture ---
type OnFinishArgs = { isDisconnect?: boolean; isAbort?: boolean };
type UseChatOptions = {
  onFinish?: (args: OnFinishArgs) => void | Promise<void>;
  onError?: (e: Error) => void;
};
let capturedUseChatOptions: UseChatOptions | null = null;
const mockResumeStream = vi.fn();
const mockSdkStop = vi.fn();
const mockSdkSendMessage = vi.fn();
const mockSetMessages = vi.fn();
let mockMessages: UIMessage[] = [];
let mockStatus: "ready" | "streaming" | "submitted" | "error" = "ready";

vi.mock("@ai-sdk/react", () => ({
  useChat: (opts: UseChatOptions) => {
    capturedUseChatOptions = opts;
    return {
      messages: mockMessages,
      sendMessage: mockSdkSendMessage,
      stop: mockSdkStop,
      status: mockStatus,
      error: undefined,
      setMessages: mockSetMessages,
      resumeStream: mockResumeStream,
    };
  },
}));

// Import after mocks
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotStream } from "../useCopilotStream";

type Args = Parameters<typeof useCopilotStream>[0];

function makeArgs(overrides: Partial<Args> = {}): Args {
  return {
    sessionId: "sess-1",
    hydratedMessages: undefined,
    hasActiveStream: false,
    refetchSession: vi.fn(async () => ({ data: undefined })),
    copilotMode: undefined,
    copilotModel: undefined,
    ...overrides,
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  mockMessages = [];
  mockStatus = "ready";
  capturedUseChatOptions = null;
  mockResumeStream.mockClear();
  mockSdkStop.mockClear();
  mockSdkSendMessage.mockClear();
  mockSetMessages.mockClear();
  mockToast.mockClear();
  mockHasActiveBackendStream.mockReturnValue(false);
  mockDisconnectSessionStream.mockClear();
  mockInvalidateQueries.mockClear();
  // Zustand stores are module singletons — wipe per-session coord state so
  // tests don't leak into each other.
  useCopilotStreamStore.getState().resetAll();
});

afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

describe("useCopilotStream — hydration/resume race (SECRT-2242)", () => {
  it("defers resume until hydration completes when hydratedMessages arrives late", () => {
    // Arrange: active backend stream, hydration NOT yet complete.
    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: undefined,
      }),
    });

    // Resume effect runs but must wait — no resumeStream fires yet.
    expect(mockResumeStream).not.toHaveBeenCalled();

    // Hydration completes: hydratedMessages becomes defined.
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [],
      }),
    );

    // pendingResumeRef should have been flushed exactly once.
    expect(mockResumeStream).toHaveBeenCalledTimes(1);
  });

  it("does not double-resume when hydration arrives after an already-queued resume", () => {
    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: undefined,
      }),
    });

    // Hydration completes once — pending resume flushes.
    rerender(makeArgs({ hasActiveStream: true, hydratedMessages: [] }));
    expect(mockResumeStream).toHaveBeenCalledTimes(1);

    // Subsequent rerender with the same hydration must not re-trigger resume
    // because hasResumedRef is now set for this session.
    rerender(makeArgs({ hasActiveStream: true, hydratedMessages: [] }));
    expect(mockResumeStream).toHaveBeenCalledTimes(1);
  });

  it("resumes immediately when hydration completed before the active-stream flag flipped on", () => {
    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: false,
        hydratedMessages: [],
      }),
    });

    expect(mockResumeStream).not.toHaveBeenCalled();

    rerender(makeArgs({ hasActiveStream: true, hydratedMessages: [] }));

    expect(mockResumeStream).toHaveBeenCalledTimes(1);
  });
});

describe("useCopilotStream — session epoch guard (SECRT-2241)", () => {
  it("reconnect timer bails out after a mid-flight session switch", async () => {
    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    });

    expect(capturedUseChatOptions?.onFinish).toBeDefined();

    // Trigger reconnect: simulate a backend-disconnect finish.
    await act(async () => {
      await capturedUseChatOptions!.onFinish!({ isDisconnect: true });
    });

    // Session switch happens BEFORE the reconnect timer fires.
    rerender(makeArgs({ sessionId: "sess-B" }));

    // Advance past the first reconnect backoff (1s) — the stale timer should
    // either have been cleared or bail via the epoch check. Either way,
    // resumeStream must not fire for the old session.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(mockResumeStream).not.toHaveBeenCalled();
  });
});

describe("useCopilotStream — forced reconnect timeout (SECRT-2241)", () => {
  it("forces UI back to idle after 30s of continuous reconnection", async () => {
    const refetchSession = vi.fn(async () => ({ data: undefined }));
    const { result } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ refetchSession }),
    });

    // Kick off reconnect via a disconnect finish.
    await act(async () => {
      await capturedUseChatOptions!.onFinish!({ isDisconnect: true });
    });

    // Initially reconnect is scheduled — but we won't let it succeed. Instead
    // advance exactly 30s so the forced-timeout callback fires.
    mockToast.mockClear();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    // Forced timeout should have fired its toast.
    const timeoutToast = mockToast.mock.calls.find(
      ([call]) => (call as { title?: string }).title === "Connection timed out",
    );
    expect(timeoutToast).toBeDefined();

    // isReconnecting should flip back to false once reconnectExhausted is set.
    expect(result.current.isReconnecting).toBe(false);
  });

  it("forced-timeout does not fire once the stream resumes (epoch/active-stream guard)", async () => {
    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs(),
    });

    await act(async () => {
      await capturedUseChatOptions!.onFinish!({ isDisconnect: true });
    });

    // Simulate the stream resuming before the 30s timeout fires by switching
    // sessions (which bumps the epoch and clears the timeout timer).
    rerender(makeArgs({ sessionId: "sess-C" }));

    mockToast.mockClear();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    const timeoutToast = mockToast.mock.calls.find(
      ([call]) => (call as { title?: string }).title === "Connection timed out",
    );
    expect(timeoutToast).toBeUndefined();
  });
});

describe("useCopilotStream — content-gated snapshot clear", () => {
  it("restores the snapshot when replay streams but no visible content arrives within the grace window", async () => {
    const trailingAssistant: UIMessage = {
      id: "hydrated-assistant",
      role: "assistant",
      parts: [{ type: "text", text: "hydrated content", state: "done" }],
    };
    mockMessages = [trailingAssistant];

    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: [trailingAssistant],
      }),
    });

    expect(mockResumeStream).toHaveBeenCalledTimes(1);

    // Our setMessages mock does not execute the updater, so
    // stripAndResume's snapshot-capture produced null. Prime the store
    // with the snapshot stripAndResume would have saved in real use.
    useCopilotStreamStore
      .getState()
      .updateCoord("sess-1", { stripSnapshot: trailingAssistant });

    // Replay begins streaming but all accumulated parts are invisible
    // (step-start only — no rendered content). Status flips to "streaming".
    mockMessages = [
      {
        id: "replay-assistant",
        role: "assistant",
        parts: [
          { type: "step-start" } as unknown as UIMessage["parts"][number],
        ],
      },
    ];
    mockStatus = "streaming";
    mockSetMessages.mockClear();
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [trailingAssistant],
      }),
    );

    // Snapshot must remain armed: no visible content yet.
    expect(
      useCopilotStreamStore.getState().getCoord("sess-1").stripSnapshot,
    ).not.toBeNull();

    // Advance past the 8s grace window.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(8_000);
    });

    // Restorer should have run: when prev contains the empty replay
    // assistant, it swaps it for the snapshot.
    const restorer = mockSetMessages.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const next = (arg as (prev: UIMessage[]) => UIMessage[])([
        {
          id: "replay-assistant",
          role: "assistant",
          parts: [
            { type: "step-start" } as unknown as UIMessage["parts"][number],
          ],
        } as UIMessage,
      ]);
      return (
        Array.isArray(next) &&
        next.length === 1 &&
        next[0].id === "hydrated-assistant"
      );
    });
    expect(restorer).toBeDefined();
  });

  it("clears the snapshot and cancels the grace timer once the replay produces visible content", async () => {
    const trailingAssistant: UIMessage = {
      id: "hydrated-assistant",
      role: "assistant",
      parts: [{ type: "text", text: "hydrated content", state: "done" }],
    };
    mockMessages = [trailingAssistant];

    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: [trailingAssistant],
      }),
    });

    expect(mockResumeStream).toHaveBeenCalledTimes(1);

    // Prime the snapshot manually (see the other test for why).
    useCopilotStreamStore
      .getState()
      .updateCoord("sess-1", { stripSnapshot: trailingAssistant });

    // Replay streams a reasoning part with actual content — this IS visible.
    mockMessages = [
      {
        id: "replay-assistant",
        role: "assistant",
        parts: [
          {
            type: "reasoning",
            text: "analyzing the question...",
          } as unknown as UIMessage["parts"][number],
        ],
      },
    ];
    mockStatus = "streaming";
    mockSetMessages.mockClear();
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [trailingAssistant],
      }),
    );

    // Snapshot is discarded because visible content has arrived.
    expect(
      useCopilotStreamStore.getState().getCoord("sess-1").stripSnapshot,
    ).toBeNull();

    // Advance past 8s — the grace timer must have been cancelled, so no
    // restore call should have been issued.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(9_000);
    });
    const restorer = mockSetMessages.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const next = (arg as (prev: UIMessage[]) => UIMessage[])([]);
      return (
        Array.isArray(next) && next.some((m) => m.id === "hydrated-assistant")
      );
    });
    expect(restorer).toBeUndefined();
  });
});

describe("useCopilotStream — resume snapshot guard", () => {
  it("restores the stripped trailing assistant if submitted→ready without streaming", async () => {
    // Arrange: an assistant bubble already hydrated into the chat.
    const trailingAssistant: UIMessage = {
      id: "hydrated-assistant",
      role: "assistant",
      parts: [{ type: "text", text: "hydrated content", state: "done" }],
    };
    mockMessages = [trailingAssistant];

    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: [trailingAssistant],
      }),
    });

    // Resume fires and strips the trailing assistant (the SDK will build
    // fresh when the backend replays from "0-0"). The SDK mock only records
    // calls — it doesn't execute the updater — so stripAndResume's own
    // snapshot capture produces null. Prime the store with the snapshot it
    // would have saved in real use.
    expect(mockResumeStream).toHaveBeenCalledTimes(1);
    useCopilotStreamStore
      .getState()
      .updateCoord("sess-1", { stripSnapshot: trailingAssistant });

    // Simulate the resume kicking the SDK into "submitted" with no chunks,
    // then dropping back to "ready" (e.g. 204 Not Found because the stream
    // already finished). The snapshot must be restored via setMessages.
    mockSetMessages.mockClear();
    mockStatus = "submitted";
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [trailingAssistant],
      }),
    );
    mockStatus = "ready";
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [trailingAssistant],
      }),
    );

    // One of the setMessages calls should be the restoration (append snapshot).
    const restorer = mockSetMessages.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const next = (arg as (prev: UIMessage[]) => UIMessage[])([]);
      return (
        Array.isArray(next) &&
        next.length === 1 &&
        next[0].id === "hydrated-assistant"
      );
    });
    expect(restorer).toBeDefined();
  });
});
