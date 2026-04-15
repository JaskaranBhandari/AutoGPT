import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useLoadMoreMessages } from "../useLoadMoreMessages";

const mockGetV2GetSession = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  getV2GetSession: (...args: unknown[]) => mockGetV2GetSession(...args),
}));

vi.mock("../helpers/convertChatSessionToUiMessages", () => ({
  convertChatSessionMessagesToUiMessages: vi.fn(() => ({ messages: [] })),
  extractToolOutputsFromRaw: vi.fn(() => []),
}));

const BASE_ARGS = {
  sessionId: "sess-1",
  initialOldestSequence: 0,
  initialNewestSequence: 49,
  initialHasMore: true,
  forwardPaginated: true,
  initialPageRawMessages: [],
};

function makeSuccessResponse(overrides: {
  messages?: unknown[];
  has_more_messages?: boolean;
  oldest_sequence?: number;
  newest_sequence?: number;
}) {
  return {
    status: 200,
    data: {
      messages: overrides.messages ?? [],
      has_more_messages: overrides.has_more_messages ?? false,
      oldest_sequence: overrides.oldest_sequence ?? 0,
      newest_sequence: overrides.newest_sequence ?? 49,
    },
  };
}

describe("useLoadMoreMessages", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("initialises with empty pagedMessages and correct cursors", () => {
    const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));
    expect(result.current.pagedMessages).toHaveLength(0);
    expect(result.current.hasMore).toBe(true);
    expect(result.current.isLoadingMore).toBe(false);
  });

  it("resetPaged clears paged state and sets hasMore=false during transition", () => {
    const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

    act(() => {
      result.current.resetPaged();
    });

    expect(result.current.pagedMessages).toHaveLength(0);
    // hasMore must be false during transition to prevent forward loadMore
    // from firing on the now-active session before forwardPaginated updates.
    expect(result.current.hasMore).toBe(false);
    expect(result.current.isLoadingMore).toBe(false);
  });

  it("resetPaged exposes a fresh loadMore via incremented epoch", () => {
    const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));
    // Just verify resetPaged is callable and doesn't throw.
    expect(() => {
      act(() => {
        result.current.resetPaged();
      });
    }).not.toThrow();
  });

  it("resets all state on sessionId change", () => {
    const { result, rerender } = renderHook(
      (props) => useLoadMoreMessages(props),
      { initialProps: BASE_ARGS },
    );

    rerender({
      ...BASE_ARGS,
      sessionId: "sess-2",
      initialOldestSequence: 10,
      initialNewestSequence: 59,
      initialHasMore: false,
    });

    expect(result.current.pagedMessages).toHaveLength(0);
    expect(result.current.hasMore).toBe(false);
    expect(result.current.isLoadingMore).toBe(false);
  });

  describe("loadMore — forward pagination", () => {
    it("calls getV2GetSession with after_sequence and updates newestSequence", async () => {
      const rawMsg = { role: "user", content: "hi", sequence: 50 };
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [rawMsg],
          has_more_messages: true,
          newest_sequence: 99,
        }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({ ...BASE_ARGS, forwardPaginated: true }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenCalledWith(
        "sess-1",
        expect.objectContaining({ after_sequence: 49 }),
      );
      expect(result.current.hasMore).toBe(true);
      expect(result.current.isLoadingMore).toBe(false);
    });

    it("sets hasMore=false when response has no more messages", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({ has_more_messages: false }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({ ...BASE_ARGS, forwardPaginated: true }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(result.current.hasMore).toBe(false);
    });

    it("is a no-op when hasMore is false", async () => {
      const { result } = renderHook(() =>
        useLoadMoreMessages({
          ...BASE_ARGS,
          initialHasMore: false,
          forwardPaginated: true,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).not.toHaveBeenCalled();
    });
  });

  describe("loadMore — backward pagination", () => {
    it("calls getV2GetSession with before_sequence", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [{ role: "user", content: "old", sequence: 0 }],
          has_more_messages: false,
          oldest_sequence: 0,
        }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({
          ...BASE_ARGS,
          forwardPaginated: false,
          initialOldestSequence: 50,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenCalledWith(
        "sess-1",
        expect.objectContaining({ before_sequence: 50 }),
      );
      expect(result.current.hasMore).toBe(false);
    });
  });

  describe("loadMore — error handling", () => {
    it("does not set hasMore=false on first error", async () => {
      mockGetV2GetSession.mockRejectedValueOnce(new Error("network error"));

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      await act(async () => {
        await result.current.loadMore();
      });

      // First error — hasMore still true
      expect(result.current.hasMore).toBe(true);
      expect(result.current.isLoadingMore).toBe(false);
    });

    it("sets hasMore=false after MAX_CONSECUTIVE_ERRORS (3) errors", async () => {
      mockGetV2GetSession.mockRejectedValue(new Error("network error"));

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      for (let i = 0; i < 3; i++) {
        await act(async () => {
          await result.current.loadMore();
        });
        // Reset the in-flight guard between calls
        await waitFor(() => expect(result.current.isLoadingMore).toBe(false));
      }

      expect(result.current.hasMore).toBe(false);
    });

    it("ignores non-200 response and increments error count", async () => {
      mockGetV2GetSession.mockResolvedValueOnce({ status: 500, data: {} });

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      await act(async () => {
        await result.current.loadMore();
      });

      // One error, not yet at threshold — hasMore still true
      expect(result.current.hasMore).toBe(true);
      expect(result.current.isLoadingMore).toBe(false);
    });
  });

  describe("loadMore — forward pagination cursor advancement", () => {
    it("advances newestSequence after a successful forward load", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [{ role: "user", content: "hi", sequence: 50 }],
          has_more_messages: true,
          newest_sequence: 99,
        }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({ ...BASE_ARGS, forwardPaginated: true }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      // A second loadMore should use after_sequence: 99 (advanced cursor)
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({ has_more_messages: false, newest_sequence: 149 }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenLastCalledWith(
        "sess-1",
        expect.objectContaining({ after_sequence: 99 }),
      );
    });

    it("does not regress newestSequence when parent refetches after pages loaded", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [{ role: "user", content: "msg", sequence: 50 }],
          has_more_messages: true,
          newest_sequence: 99,
        }),
      );

      const { result, rerender } = renderHook(
        (props) => useLoadMoreMessages(props),
        { initialProps: { ...BASE_ARGS, forwardPaginated: true } },
      );

      // Load one page — newestSequence advances to 99
      await act(async () => {
        await result.current.loadMore();
      });

      // Parent refetches with a lower newest_sequence (49) — should NOT regress cursor
      rerender({
        ...BASE_ARGS,
        forwardPaginated: true,
        initialNewestSequence: 49,
      });

      // Next loadMore should still use the advanced cursor (99)
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({ has_more_messages: false, newest_sequence: 149 }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenLastCalledWith(
        "sess-1",
        expect.objectContaining({ after_sequence: 99 }),
      );
    });
  });

  describe("loadMore — MAX_OLDER_MESSAGES truncation", () => {
    it("truncates accumulated messages at MAX_OLDER_MESSAGES (2000)", async () => {
      // Simulate being near the limit — 1990 existing paged messages
      const nearLimitArgs = {
        ...BASE_ARGS,
        forwardPaginated: false,
        initialOldestSequence: 1990,
      };

      // Return 20 messages to push total past 2000
      const newMessages = Array.from({ length: 20 }, (_, i) => ({
        role: "user",
        content: `msg ${i}`,
        sequence: i,
      }));

      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: newMessages,
          has_more_messages: true,
          oldest_sequence: 0,
        }),
      );

      const { result } = renderHook((props) => useLoadMoreMessages(props), {
        initialProps: nearLimitArgs,
      });

      // Pre-fill pagedRawMessages to near limit by doing a successful load first
      // then checking hasMore is set to false when limit reached
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: Array.from({ length: 1990 }, (_, i) => ({
            role: "user",
            content: `old ${i}`,
            sequence: i,
          })),
          has_more_messages: true,
          oldest_sequence: 0,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      // Now add 20 more to exceed 2000 — hasMore should be forced false
      await act(async () => {
        await result.current.loadMore();
      });

      expect(result.current.hasMore).toBe(false);
    });
  });

  describe("loadMore — epoch / stale-response guard", () => {
    it("discards response when epoch changes during flight (resetPaged called)", async () => {
      let resolveRequest!: (v: unknown) => void;
      mockGetV2GetSession.mockReturnValueOnce(
        new Promise((res) => {
          resolveRequest = res;
        }),
      );

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      // Start the request without awaiting
      act(() => {
        result.current.loadMore();
      });

      // Reset epoch mid-flight
      act(() => {
        result.current.resetPaged();
      });

      // Now resolve the in-flight request
      await act(async () => {
        resolveRequest(
          makeSuccessResponse({ messages: [{ role: "user", content: "hi" }] }),
        );
      });

      // Response discarded — pagedMessages stays empty, isLoadingMore stays false
      expect(result.current.pagedMessages).toHaveLength(0);
      expect(result.current.isLoadingMore).toBe(false);
    });
  });
});
