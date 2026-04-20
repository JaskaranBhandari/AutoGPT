import {
  getGetV2GetCopilotUsageQueryKey,
  getGetV2GetSessionQueryKey,
  postV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { environment } from "@/services/environment";
import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import type { FileUIPart, UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { useCopilotStreamStore } from "./copilotStreamStore";
import {
  getCopilotAuthHeaders,
  deduplicateMessages,
  extractSendMessageText,
  hasActiveBackendStream,
  hasVisibleAssistantContent,
  resolveInProgressTools,
  getSendSuppressionReason,
  disconnectSessionStream,
} from "./helpers";
import type { CopilotLlmModel, CopilotMode } from "./store";
import { useHydrateOnStreamEnd } from "./useHydrateOnStreamEnd";

const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_ATTEMPTS = 3;

/** Minimum time the page must have been hidden to trigger a wake re-sync. */
const WAKE_RESYNC_THRESHOLD_MS = 30_000;

/** Max time (ms) the UI can stay in "reconnecting" state before forcing idle. */
const RECONNECT_MAX_DURATION_MS = 30_000;

/**
 * Delay after a clean stream close before refetching the session to check
 * whether the backend executor is still running. Without this, the refetch
 * races with the backend clearing `active_stream` and often reads a stale
 * `active_stream=true`, triggering unnecessary reconnect cycles.
 */
const FINISH_REFETCH_SETTLE_MS = 500;

/**
 * Time to wait after resumeStream() before concluding that the replay
 * produced no chunks. When exceeded with status still "submitted", we
 * restore the stripped assistant snapshot so the user sees the hydrated
 * content instead of an indefinite "Thinking...".
 */
const RESUME_NO_CHUNK_GRACE_MS = 8_000;

interface UseCopilotStreamArgs {
  sessionId: string | null;
  hydratedMessages: UIMessage[] | undefined;
  hasActiveStream: boolean;
  refetchSession: () => Promise<{ data?: unknown }>;
  /** Autopilot mode to use for requests. `undefined` = let backend decide via feature flags. */
  copilotMode: CopilotMode | undefined;
  /** Model tier override. `undefined` = let backend decide. */
  copilotModel: CopilotLlmModel | undefined;
}

export function useCopilotStream({
  sessionId,
  hydratedMessages,
  hasActiveStream,
  refetchSession,
  copilotMode,
  copilotModel,
}: UseCopilotStreamArgs) {
  const queryClient = useQueryClient();
  const [rateLimitMessage, setRateLimitMessage] = useState<string | null>(null);
  function dismissRateLimit() {
    setRateLimitMessage(null);
  }
  // Use refs for copilotMode and copilotModel so the transport closure always reads
  // the latest value without recreating the DefaultChatTransport (which would
  // reset useChat's internal Chat instance and break mid-session streaming).
  const copilotModeRef = useRef(copilotMode);
  copilotModeRef.current = copilotMode;
  const copilotModelRef = useRef(copilotModel);
  copilotModelRef.current = copilotModel;

  // Connect directly to the Python backend for SSE, bypassing the Next.js
  // serverless proxy. This eliminates the Vercel 800s function timeout that
  // was the primary cause of stream disconnections on long-running tasks.
  // Auth uses the same server-action token pattern as the WebSocket connection.
  const transport = useMemo(
    () =>
      sessionId
        ? new DefaultChatTransport({
            api: `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`,
            prepareSendMessagesRequest: async ({ messages }) => {
              const last = messages[messages.length - 1];
              // Extract file_ids from FileUIPart entries on the message
              const fileIds = last.parts
                ?.filter((p): p is FileUIPart => p.type === "file")
                .map((p) => {
                  // URL is like /api/proxy/api/workspace/files/{id}/download
                  const match = p.url.match(/\/workspace\/files\/([^/]+)\//);
                  return match?.[1];
                })
                .filter(Boolean) as string[] | undefined;
              return {
                body: {
                  message: (
                    last.parts?.map((p) => (p.type === "text" ? p.text : "")) ??
                    []
                  ).join(""),
                  is_user_message: last.role === "user",
                  context: null,
                  file_ids: fileIds && fileIds.length > 0 ? fileIds : null,
                  mode: copilotModeRef.current ?? null,
                  model: copilotModelRef.current ?? null,
                },
                headers: await getCopilotAuthHeaders(),
              };
            },
            prepareReconnectToStreamRequest: async () => ({
              api: `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`,
              headers: await getCopilotAuthHeaders(),
            }),
          })
        : null,
    [sessionId],
  );

  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const reconnectTimeoutTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const resumeGraceTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const lastHiddenAtRef = useRef(Date.now());
  const sessionEpochRef = useRef(0);
  // Reactive flag that drives the isReconnecting UI state.
  const [isReconnectScheduled, setIsReconnectScheduled] = useState(false);
  const [reconnectExhausted, setReconnectExhausted] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  // Synchronous flag read inside SDK callbacks — kept as a ref so callbacks
  // don't have to trigger re-renders to observe changes.
  const isUserStoppingRef = useRef(false);

  function handleReconnect(sid: string) {
    if (!sid) return;
    const coord = useCopilotStreamStore.getState().getCoord(sid);
    if (coord.reconnectScheduled) return;

    const nextAttempt = coord.reconnectAttempts + 1;
    if (nextAttempt > RECONNECT_MAX_ATTEMPTS) {
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
      setReconnectExhausted(true);
      toast({
        title: "Connection lost",
        description: "Unable to reconnect. Please refresh the page.",
        variant: "destructive",
      });
      return;
    }

    // Track when reconnection first started for the forced timeout.
    if (coord.reconnectStartedAt === null) {
      useCopilotStreamStore
        .getState()
        .updateCoord(sid, { reconnectStartedAt: Date.now() });
      // Schedule a forced timeout — if reconnecting takes longer than
      // RECONNECT_MAX_DURATION_MS, force the UI back to idle.
      clearTimeout(reconnectTimeoutTimerRef.current);
      const capturedEpoch = sessionEpochRef.current;
      reconnectTimeoutTimerRef.current = setTimeout(() => {
        if (sessionEpochRef.current !== capturedEpoch) return;
        // Cancel the pending reconnect timer so it can't fire stripAndResume()
        // after the UI has been forced to idle — otherwise we'd end up in an
        // inconsistent state (reconnectExhausted=true + a fresh stream).
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = undefined;
        useCopilotStreamStore.getState().updateCoord(sid, {
          reconnectScheduled: false,
          reconnectStartedAt: null,
        });
        setIsReconnectScheduled(false);
        setReconnectExhausted(true);
        toast({
          title: "Connection timed out",
          description:
            "AutoPilot may still be working. Refresh to check for updates.",
          variant: "destructive",
        });
      }, RECONNECT_MAX_DURATION_MS);
    }

    useCopilotStreamStore.getState().updateCoord(sid, {
      reconnectScheduled: true,
      reconnectAttempts: nextAttempt,
    });
    setIsReconnectScheduled(true);

    if (!coord.hasShownDisconnectToast) {
      useCopilotStreamStore
        .getState()
        .updateCoord(sid, { hasShownDisconnectToast: true });
      toast({ title: "Connection lost", description: "Reconnecting..." });
    }

    const delay = RECONNECT_BASE_DELAY_MS * 2 ** (nextAttempt - 1);
    const capturedEpoch = sessionEpochRef.current;

    reconnectTimerRef.current = setTimeout(() => {
      // Bail if the session switched while the timer was pending.
      if (sessionEpochRef.current !== capturedEpoch) return;

      useCopilotStreamStore.getState().updateCoord(sid, {
        reconnectScheduled: false,
        // Mark this session as resumed so a deferred page-load resume
        // (queued in the store while hydration was in-flight) can't
        // double-fire resumeStream() once hydration completes.
        hasResumed: true,
      });
      setIsReconnectScheduled(false);
      stripAndResume(sid);
    }, delay);
  }

  /**
   * Strip the trailing assistant message (so the SDK doesn't duplicate parts
   * when the backend replays from "0-0") and trigger the resume GET. The
   * stripped message is saved in the store; the status-transition effect
   * either confirms the replay arrived (discards snapshot) or concludes it
   * never will (restores snapshot so the user isn't stuck on "Thinking...").
   */
  function stripAndResume(sid: string) {
    let snapshot: UIMessage | null = null;
    setMessages((prev) => {
      if (prev.length > 0 && prev[prev.length - 1].role === "assistant") {
        snapshot = prev[prev.length - 1];
        return prev.slice(0, -1);
      }
      return prev;
    });
    useCopilotStreamStore
      .getState()
      .updateCoord(sid, { stripSnapshot: snapshot });

    // Start a grace timer: if no chunks arrive in time, restore the snapshot.
    // The status-transition effect clears this timer if streaming begins.
    clearTimeout(resumeGraceTimerRef.current);
    const capturedEpoch = sessionEpochRef.current;
    resumeGraceTimerRef.current = setTimeout(() => {
      if (sessionEpochRef.current !== capturedEpoch) return;
      restoreSnapshotIfPresent(sid);
    }, RESUME_NO_CHUNK_GRACE_MS);

    resumeStreamRef.current();
  }

  function restoreSnapshotIfPresent(sid: string) {
    const { stripSnapshot } = useCopilotStreamStore.getState().getCoord(sid);
    if (!stripSnapshot) return;
    useCopilotStreamStore.getState().updateCoord(sid, { stripSnapshot: null });
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role !== "assistant") {
        return [...prev, stripSnapshot];
      }
      // Trailing assistant is the replay. If it has any rendered content we
      // leave it alone (the replay is working). If it's still empty after
      // the grace window — e.g. status went to "streaming" but only empty
      // reasoning-start / step-start parts arrived — swap it for the
      // snapshot so the user sees the hydrated content instead of a stuck
      // "Thinking..." bubble.
      if (hasVisibleAssistantContent(prev)) return prev;
      return [...prev.slice(0, -1), stripSnapshot];
    });
  }

  const {
    messages: rawMessages,
    sendMessage: sdkSendMessage,
    stop: sdkStop,
    status,
    error,
    setMessages,
    resumeStream,
  } = useChat({
    id: sessionId ?? undefined,
    transport: transport ?? undefined,
    onFinish: async ({ isDisconnect, isAbort }) => {
      if (isAbort || !sessionId) return;
      // User-initiated stops should not trigger reconnection.
      if (isUserStoppingRef.current) return;

      // The AI SDK rarely sets isDisconnect — treat ANY non-user-initiated
      // finish as a potential disconnect when the backend stream is active.
      if (isDisconnect) {
        handleReconnect(sessionId);
        return;
      }

      // Check if backend executor is still running after clean close.
      await new Promise((r) => setTimeout(r, FINISH_REFETCH_SETTLE_MS));
      const result = await refetchSession();
      if (hasActiveBackendStream(result)) {
        handleReconnect(sessionId);
      }
    },
    onError: (error) => {
      if (!sessionId) return;

      // Detect rate limit (429) responses and show reset time to the user.
      // The SDK throws a plain Error whose message is the raw response body
      // (FastAPI returns {"detail": "...usage limit..."} for 429s).
      let errorDetail: string = error.message;
      try {
        const parsed = JSON.parse(error.message) as unknown;
        if (
          typeof parsed === "object" &&
          parsed !== null &&
          "detail" in parsed &&
          typeof (parsed as { detail: unknown }).detail === "string"
        ) {
          errorDetail = (parsed as { detail: string }).detail;
        }
      } catch {
        // Not JSON — use message as-is
      }
      const isRateLimited = errorDetail.toLowerCase().includes("usage limit");
      if (isRateLimited) {
        setRateLimitMessage(
          errorDetail ||
            "You've reached your usage limit. Please try again later.",
        );
        return;
      }

      // Detect authentication failures (from getCopilotAuthHeaders or 401 responses)
      const isAuthError =
        errorDetail.includes("Authentication failed") ||
        errorDetail.includes("Unauthorized") ||
        errorDetail.includes("Not authenticated") ||
        errorDetail.toLowerCase().includes("401");
      if (isAuthError) {
        toast({
          title: "Authentication error",
          description: "Your session may have expired. Please sign in again.",
          variant: "destructive",
        });
        return;
      }

      // Reconnect on network errors or transient API errors so the
      // persisted retryable-error marker is loaded and the "Try Again"
      // button appears.  Without this, transient errors only show in the
      // onError callback (where StreamError strips the retryable prefix).
      if (isUserStoppingRef.current) return;
      const isNetworkError =
        error.name === "TypeError" || error.name === "AbortError";
      const isTransientApiError = errorDetail.includes(
        "connection interrupted",
      );
      if (isNetworkError || isTransientApiError) {
        handleReconnect(sessionId);
      }
    },
  });

  // Keep stable refs to sdkStop and resumeStream so that async callbacks
  // (session-switch cleanup, wake re-sync, reconnect timer) always call the
  // latest version without stale-closure bugs.
  const sdkStopRef = useRef(sdkStop);
  sdkStopRef.current = sdkStop;
  const resumeStreamRef = useRef(resumeStream);
  resumeStreamRef.current = resumeStream;

  // Wrap sdkSendMessage to guard against re-sending the user message during a
  // reconnect cycle. If the session already has the message (i.e. we are in a
  // reconnect/resume flow), only GET-resume is safe — never re-POST.
  const sendMessage: typeof sdkSendMessage = async (...args) => {
    const text = extractSendMessageText(args[0]);
    const sid = sessionId;
    const coord = sid ? useCopilotStreamStore.getState().getCoord(sid) : null;

    const suppressReason = getSendSuppressionReason({
      text,
      isReconnectScheduled: coord?.reconnectScheduled ?? false,
      lastSubmittedText: coord?.lastSubmittedMessageText ?? null,
      messages: rawMessages,
    });

    if (suppressReason === "reconnecting") {
      // The ref flips to ``true`` synchronously while the React state that
      // drives the UI's disabled state only updates on the next render, so
      // the user may have clicked send against a still-enabled input. Tell
      // them their message wasn't dropped silently.
      toast({
        title: "Reconnecting",
        description: "Wait for the connection to resume before sending.",
      });
      return;
    }
    if (suppressReason === "duplicate") return;

    if (sid) {
      useCopilotStreamStore
        .getState()
        .updateCoord(sid, { lastSubmittedMessageText: text });
    }
    return sdkSendMessage(...args);
  };

  // Deduplicate messages continuously to prevent duplicates when resuming streams
  const messages = useMemo(
    () => deduplicateMessages(rawMessages),
    [rawMessages],
  );

  // Wrap AI SDK's stop() to also cancel the backend executor task.
  // sdkStop() aborts the SSE fetch instantly (UI feedback), then we fire
  // the cancel API to actually stop the executor and wait for confirmation.
  async function stop() {
    isUserStoppingRef.current = true;
    sdkStop();
    // Resolve pending tool calls and inject a cancellation marker so the UI
    // shows "You manually stopped this chat" immediately (the backend writes
    // the same marker to the DB, but the SSE connection is already aborted).
    // Marker must match COPILOT_ERROR_PREFIX in ChatMessagesContainer/helpers.ts.
    setMessages((prev) => {
      const resolved = resolveInProgressTools(prev, "cancelled");
      const last = resolved[resolved.length - 1];
      if (last?.role === "assistant") {
        return [
          ...resolved.slice(0, -1),
          {
            ...last,
            parts: [
              ...last.parts,
              {
                type: "text" as const,
                text: "[__COPILOT_ERROR_f7a1__] Operation cancelled",
              },
            ],
          },
        ];
      }
      return resolved;
    });

    if (!sessionId) return;
    try {
      const res = await postV2CancelSessionTask(sessionId);
      if (
        res.status === 200 &&
        "reason" in res.data &&
        res.data.reason === "cancel_published_not_confirmed"
      ) {
        toast({
          title: "Stop may take a moment",
          description:
            "The cancel was sent but not yet confirmed. The task should stop shortly.",
        });
      }
    } catch {
      toast({
        title: "Could not stop the task",
        description: "The task may still be running in the background.",
        variant: "destructive",
      });
    }
  }

  // Keep a ref to sessionId so the async wake handler can detect staleness.
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  // ---------------------------------------------------------------------------
  // Wake detection: when the page becomes visible after being hidden for >30s
  // (device sleep, tab backgrounded for a long time), refetch the session to
  // pick up any messages the backend produced while the SSE was dead.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    async function handleWakeResync() {
      const sid = sessionIdRef.current;
      if (!sid) return;

      const elapsed = Date.now() - lastHiddenAtRef.current;
      lastHiddenAtRef.current = Date.now();

      if (document.visibilityState !== "visible") return;
      if (elapsed < WAKE_RESYNC_THRESHOLD_MS) return;

      setIsSyncing(true);
      try {
        const result = await refetchSession();
        // Bail out if the session changed while the refetch was in flight.
        if (sessionIdRef.current !== sid) return;

        if (hasActiveBackendStream(result)) {
          // Stream is still running — resume SSE to pick up live chunks.
          // stripAndResume handles the snapshot save/restore dance.
          stripAndResume(sid);
        }
        // If !backendActive, the refetch will update hydratedMessages via
        // React Query, and the hydration effect below will merge them in.
      } catch (err) {
        console.warn("[copilot] wake re-sync failed", err);
      } finally {
        setIsSyncing(false);
      }
    }

    function onVisibilityChange() {
      if (document.visibilityState === "hidden") {
        lastHiddenAtRef.current = Date.now();
      } else {
        handleWakeResync();
      }
    }

    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [refetchSession, setMessages]);

  // After-stream hydration — force-replace AI-SDK state with the DB's view
  // once React Query has actually refetched, then keep length-gated top-ups
  // working for pagination. See useHydrateOnStreamEnd for the timing dance.
  useHydrateOnStreamEnd({
    status,
    hydratedMessages,
    isReconnectScheduled,
    setMessages,
  });

  // Mark hydrateCompleted in the store whenever the hydration gate has
  // effectively run (hydrated data is present and we're not mid-stream), and
  // flush any `pendingResume` that was deferred while hydration was still
  // pending on the active session. Kept as a separate effect so the
  // useHydrateOnStreamEnd hook stays focused on the AI-SDK state sync.
  useEffect(() => {
    if (!sessionId) return;
    if (!hydratedMessages) return;
    if (status === "streaming" || status === "submitted") return;
    if (isReconnectScheduled) return;

    useCopilotStreamStore
      .getState()
      .updateCoord(sessionId, { hydrateCompleted: true });
    const { pendingResume } = useCopilotStreamStore
      .getState()
      .getCoord(sessionId);
    if (pendingResume) {
      useCopilotStreamStore
        .getState()
        .updateCoord(sessionId, { pendingResume: null });
      pendingResume();
    }
  }, [sessionId, hydratedMessages, status, isReconnectScheduled]);

  // Clean up state on session switch.
  // Abort the old stream's in-flight fetch and tell the backend to release
  // its XREAD listeners immediately (fire-and-forget). Drop the previous
  // session's per-session store record so a future revisit starts fresh.
  const prevStreamSessionRef = useRef(sessionId);
  useEffect(() => {
    const prevSid = prevStreamSessionRef.current;
    prevStreamSessionRef.current = sessionId;

    // Bump epoch so stale async callbacks from the old session bail out.
    sessionEpochRef.current += 1;
    const currentEpoch = sessionEpochRef.current;
    useCopilotStreamStore.getState().setActiveSession(sessionId);

    const isSwitching = Boolean(prevSid && prevSid !== sessionId);
    if (isSwitching) {
      // Mark BEFORE stopping so the old stream's async onError (which fires
      // after the abort) sees the flag and short-circuits the reconnect path.
      isUserStoppingRef.current = true;
      sdkStopRef.current();
      disconnectSessionStream(prevSid!);
      // Drop the previous session's coord so a future visit starts fresh
      // (hasResumed=false, counters=0) rather than inheriting stale state.
      useCopilotStreamStore.getState().clearCoord(prevSid!);
      // Schedule the reset as a task (not a microtask) so it runs AFTER the
      // aborted fetch's onError has fired — but verify the epoch hasn't
      // changed again (rapid session switches).
      setTimeout(() => {
        if (sessionEpochRef.current === currentEpoch) {
          isUserStoppingRef.current = false;
        }
      }, 0);
    } else {
      isUserStoppingRef.current = false;
    }

    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = undefined;
    clearTimeout(reconnectTimeoutTimerRef.current);
    reconnectTimeoutTimerRef.current = undefined;
    clearTimeout(resumeGraceTimerRef.current);
    resumeGraceTimerRef.current = undefined;
    setIsReconnectScheduled(false);
    setRateLimitMessage(null);
    setReconnectExhausted(false);
    setIsSyncing(false);
    return () => {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = undefined;
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
      clearTimeout(resumeGraceTimerRef.current);
      resumeGraceTimerRef.current = undefined;
    };
  }, [sessionId]);

  // Invalidate session cache when stream completes; also manages the
  // snapshot guard that restores stripped content if the replay never
  // produced chunks.
  const prevStatusRef = useRef(status);
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    const wasActive = prev === "streaming" || prev === "submitted";
    const isNowActive = status === "streaming" || status === "submitted";
    const isIdle = status === "ready" || status === "error";

    // Clear the forced reconnect timeout as soon as the stream resumes —
    // otherwise the stale 30s timer can fire mid-stream and show a
    // "timed out" toast even though reconnection succeeded.
    if (
      isNowActive &&
      sessionId &&
      useCopilotStreamStore.getState().getCoord(sessionId)
        .reconnectStartedAt !== null
    ) {
      useCopilotStreamStore
        .getState()
        .updateCoord(sessionId, { reconnectStartedAt: null });
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
    }

    // Resume finished without ever streaming (e.g. 204 Not Found because
    // the backend stream finished between REST fetch and resume GET).
    // Restore the stripped snapshot so the user sees the hydrated content
    // instead of an empty bubble.
    if (prev === "submitted" && status === "ready" && sessionId) {
      restoreSnapshotIfPresent(sessionId);
    }

    if (wasActive && isIdle && sessionId && !isReconnectScheduled) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
      queryClient.invalidateQueries({
        queryKey: getGetV2GetCopilotUsageQueryKey(),
      });
      if (status === "ready") {
        useCopilotStreamStore.getState().updateCoord(sessionId, {
          reconnectAttempts: 0,
          hasShownDisconnectToast: false,
          reconnectStartedAt: null,
        });
        clearTimeout(reconnectTimeoutTimerRef.current);
        reconnectTimeoutTimerRef.current = undefined;
        // Intentionally NOT clearing lastSubmittedMessageText here: keeping
        // the last submitted text prevents getSendSuppressionReason from
        // allowing a duplicate POST of the same message immediately after a
        // successful turn (the "duplicate" branch checks both the store and
        // the visible last user message, so legitimate re-sends after a
        // different reply are still allowed).
        setReconnectExhausted(false);
      }
    }
  }, [status, sessionId, queryClient, isReconnectScheduled]);

  // Discard the resume snapshot (and cancel the 8s grace timer) as soon as
  // the replay has put something visible on screen. We do NOT gate on
  // status === "streaming" alone because Perplexity deep-research streams
  // empty reasoning-start / step-start chunks for minutes before any
  // rendered content arrives — during which the Thinking-bubble would
  // otherwise stay stuck and the grace timer restore would never fire.
  useEffect(() => {
    if (!sessionId) return;
    const { stripSnapshot } = useCopilotStreamStore
      .getState()
      .getCoord(sessionId);
    if (!stripSnapshot) return;
    if (!hasVisibleAssistantContent(rawMessages)) return;
    clearTimeout(resumeGraceTimerRef.current);
    resumeGraceTimerRef.current = undefined;
    useCopilotStreamStore
      .getState()
      .updateCoord(sessionId, { stripSnapshot: null });
  }, [rawMessages, sessionId]);

  // Resume an active stream AFTER hydration completes.
  // IMPORTANT: Only runs when page loads with existing active stream (reconnection).
  // Does NOT run when new streams start during active conversation.
  // Gated on per-session hydrateCompleted to prevent racing with the
  // hydration effect.
  useEffect(() => {
    if (!sessionId) return;
    if (!hasActiveStream) return;
    if (!hydratedMessages) return;

    // Never resume if currently streaming
    if (status === "streaming" || status === "submitted") return;

    const coord = useCopilotStreamStore.getState().getCoord(sessionId);
    // Only resume once per session
    if (coord.hasResumed) return;

    // Don't resume a stream the user just cancelled
    if (isUserStoppingRef.current) return;

    const capturedEpoch = sessionEpochRef.current;
    function doResume() {
      if (sessionEpochRef.current !== capturedEpoch) return;
      if (!sessionId) return;
      const latest = useCopilotStreamStore.getState().getCoord(sessionId);
      if (latest.hasResumed) return;
      if (isUserStoppingRef.current) return;

      useCopilotStreamStore
        .getState()
        .updateCoord(sessionId, { hasResumed: true });
      stripAndResume(sessionId);
    }

    // Wait for hydration to complete before resuming to prevent
    // the two effects from racing (duplicate messages / missing content).
    if (!coord.hydrateCompleted) {
      useCopilotStreamStore
        .getState()
        .updateCoord(sessionId, { pendingResume: doResume });
      return;
    }

    doResume();
  }, [sessionId, hasActiveStream, hydratedMessages, status, setMessages]);

  // Clear messages when session is null
  useEffect(() => {
    if (!sessionId) setMessages([]);
  }, [sessionId, setMessages]);

  // Reset the user-stop flag once the backend confirms the stream is no
  // longer active — this prevents the flag from staying stale forever.
  useEffect(() => {
    if (!hasActiveStream && isUserStoppingRef.current) {
      isUserStoppingRef.current = false;
    }
  }, [hasActiveStream]);

  // True while reconnecting or backend has active stream but we haven't connected yet.
  // Suppressed when the user explicitly stopped or when all reconnect attempts
  // are exhausted — the backend may be slow to clear active_stream but the UI
  // should remain responsive.
  const isReconnecting =
    !isUserStoppingRef.current &&
    !reconnectExhausted &&
    (isReconnectScheduled ||
      (hasActiveStream && status !== "streaming" && status !== "submitted"));

  return {
    messages,
    setMessages,
    sendMessage,
    stop,
    status,
    error: isReconnecting || isUserStoppingRef.current ? undefined : error,
    isReconnecting,
    isSyncing,
    isUserStoppingRef,
    rateLimitMessage,
    dismissRateLimit,
  };
}
