import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  convertChatSessionMessagesToUiMessages,
  extractToolOutputsFromRaw,
} from "./helpers/convertChatSessionToUiMessages";

interface UseLoadMoreMessagesArgs {
  sessionId: string | null;
  initialOldestSequence: number | null;
  initialNewestSequence: number | null;
  initialHasMore: boolean;
  /** True when the initial page was loaded from sequence 0 forward (completed
   *  sessions). False when loaded newest-first (active sessions). */
  forwardPaginated: boolean;
  /** Raw messages from the initial page, used for cross-page tool output matching. */
  initialPageRawMessages: unknown[];
}

const MAX_CONSECUTIVE_ERRORS = 3;
const MAX_OLDER_MESSAGES = 2000;

export function useLoadMoreMessages({
  sessionId,
  initialOldestSequence,
  initialNewestSequence,
  initialHasMore,
  forwardPaginated,
  initialPageRawMessages,
}: UseLoadMoreMessagesArgs) {
  // Accumulated raw messages from all extra pages (ascending order).
  // Re-converting them all together ensures tool outputs are matched across
  // inter-page boundaries.
  const [pagedRawMessages, setPagedRawMessages] = useState<unknown[]>([]);
  const [oldestSequence, setOldestSequence] = useState<number | null>(
    initialOldestSequence,
  );
  const [newestSequence, setNewestSequence] = useState<number | null>(
    initialNewestSequence,
  );
  const [hasMore, setHasMore] = useState(initialHasMore);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const isLoadingMoreRef = useRef(false);
  const consecutiveErrorsRef = useRef(0);
  // Epoch counter to discard stale loadMore responses after a reset
  const epochRef = useRef(0);

  // Track the sessionId and initial cursor to reset state on change
  const prevSessionIdRef = useRef(sessionId);
  const prevInitialOldestRef = useRef(initialOldestSequence);

  // Sync initial values from parent when they change
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      // Session changed — full reset
      prevSessionIdRef.current = sessionId;
      prevInitialOldestRef.current = initialOldestSequence;
      setPagedRawMessages([]);
      setOldestSequence(initialOldestSequence);
      setNewestSequence(initialNewestSequence);
      setHasMore(initialHasMore);
      setIsLoadingMore(false);
      isLoadingMoreRef.current = false;
      consecutiveErrorsRef.current = 0;
      epochRef.current += 1;
    } else if (
      prevInitialOldestRef.current !== initialOldestSequence &&
      pagedRawMessages.length > 0
    ) {
      // Same session but initial window shifted (e.g. new messages arrived) —
      // clear paged state to avoid gaps/duplicates
      prevInitialOldestRef.current = initialOldestSequence;
      setPagedRawMessages([]);
      setOldestSequence(initialOldestSequence);
      setNewestSequence(initialNewestSequence);
      setHasMore(initialHasMore);
      setIsLoadingMore(false);
      isLoadingMoreRef.current = false;
      consecutiveErrorsRef.current = 0;
      epochRef.current += 1;
    } else {
      // Update from parent when initial data changes (e.g. refetch)
      prevInitialOldestRef.current = initialOldestSequence;
      setOldestSequence(initialOldestSequence);
      // Only regress the forward cursor if we haven't paged ahead yet —
      // otherwise a parent refetch would reset a cursor we already advanced.
      setNewestSequence((prev) =>
        prev !== null && prev > (initialNewestSequence ?? -1)
          ? prev
          : initialNewestSequence,
      );
      setHasMore(initialHasMore);
    }
  }, [sessionId, initialOldestSequence, initialNewestSequence, initialHasMore]);

  // Convert all accumulated raw messages in one pass so tool outputs
  // are matched across inter-page boundaries.
  // For backward pagination: initial page tool outputs are included via
  // extraToolOutputs to handle the boundary between the last older page and
  // the initial/streaming page.
  const pagedMessages: UIMessage<unknown, UIDataTypes, UITools>[] =
    useMemo(() => {
      if (!sessionId || pagedRawMessages.length === 0) return [];
      const extraToolOutputs =
        !forwardPaginated && initialPageRawMessages.length > 0
          ? extractToolOutputsFromRaw(initialPageRawMessages)
          : undefined;
      return convertChatSessionMessagesToUiMessages(
        sessionId,
        pagedRawMessages,
        { isComplete: true, extraToolOutputs },
      ).messages;
    }, [sessionId, pagedRawMessages, initialPageRawMessages, forwardPaginated]);

  async function loadMore() {
    if (!sessionId || !hasMore || isLoadingMoreRef.current) return;

    const cursor = forwardPaginated ? newestSequence : oldestSequence;
    if (cursor === null) return;

    const requestEpoch = epochRef.current;
    isLoadingMoreRef.current = true;
    setIsLoadingMore(true);
    try {
      const params = forwardPaginated
        ? { limit: 50, after_sequence: cursor }
        : { limit: 50, before_sequence: cursor };
      const response = await getV2GetSession(sessionId, params);

      // Discard response if session/pagination was reset while awaiting
      if (epochRef.current !== requestEpoch) return;

      if (response.status !== 200) {
        consecutiveErrorsRef.current += 1;
        console.warn(
          `[loadMore] Failed to load messages (status=${response.status}, attempt=${consecutiveErrorsRef.current})`,
        );
        if (consecutiveErrorsRef.current >= MAX_CONSECUTIVE_ERRORS) {
          setHasMore(false);
        }
        return;
      }

      consecutiveErrorsRef.current = 0;

      const newRaw = (response.data.messages ?? []) as unknown[];
      setPagedRawMessages((prev) => {
        // Forward: append to end. Backward: prepend to start.
        const merged = forwardPaginated
          ? [...prev, ...newRaw]
          : [...newRaw, ...prev];
        if (merged.length > MAX_OLDER_MESSAGES) {
          return merged.slice(merged.length - MAX_OLDER_MESSAGES);
        }
        return merged;
      });

      if (forwardPaginated) {
        setNewestSequence(response.data.newest_sequence ?? null);
      } else {
        setOldestSequence(response.data.oldest_sequence ?? null);
      }

      if (newRaw.length + pagedRawMessages.length >= MAX_OLDER_MESSAGES) {
        setHasMore(false);
      } else {
        setHasMore(!!response.data.has_more_messages);
      }
    } catch (error) {
      if (epochRef.current !== requestEpoch) return;
      consecutiveErrorsRef.current += 1;
      console.warn("[loadMore] Network error:", error);
      if (consecutiveErrorsRef.current >= MAX_CONSECUTIVE_ERRORS) {
        setHasMore(false);
      }
    } finally {
      if (epochRef.current === requestEpoch) {
        isLoadingMoreRef.current = false;
        setIsLoadingMore(false);
      }
    }
  }

  return { pagedMessages, hasMore, isLoadingMore, loadMore };
}
