"use client";

import {
  getGetV2ListSessionsQueryKey,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
  usePatchV2UpdateSessionTitle,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { toast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  CheckCircle,
  CircleNotch,
  PencilSimple,
  Trash,
  X,
} from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useIsMobile } from "@/hooks/use-mobile";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";

export function ChatSessionList() {
  const isMobile = useIsMobile();
  const pathname = usePathname();
  const router = useRouter();
  const isCopilotPage = pathname === "/" || pathname.startsWith("/copilot");
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const activeSessionId = isCopilotPage ? sessionId : null;
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const [hoveredSessionId, setHoveredSessionId] = useState<string | null>(null);
  const {
    sessionToDelete,
    setSessionToDelete,
    completedSessionIDs,
    clearCompletedSession,
  } = useCopilotUIStore();

  const queryClient = useQueryClient();

  const {
    data: sessionsResponse,
    isLoading: isLoadingSessions,
    isError: isSessionsError,
  } = useGetV2ListSessions(
    { limit: 50 },
    { query: { refetchInterval: 10_000 } },
  );

  const { mutate: deleteSession, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          });
          if (sessionToDelete?.id === sessionId) {
            setSessionId(null);
          }
          setSessionToDelete(null);
        },
        onError: (error: unknown) => {
          toast({
            title: "Failed to delete chat",
            description:
              error instanceof Error ? error.message : "An error occurred",
            variant: "destructive",
          });
          setSessionToDelete(null);
        },
      },
    });

  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  const { mutate: renameSession, isPending: isRenaming } =
    usePatchV2UpdateSessionTitle({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          });
          setEditingSessionId(null);
        },
        onError: (error: unknown) => {
          toast({
            title: "Failed to rename chat",
            description:
              error instanceof Error ? error.message : "An error occurred",
            variant: "destructive",
          });
          setEditingSessionId(null);
        },
      },
    });

  useEffect(() => {
    if (editingSessionId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [editingSessionId]);

  useEffect(() => {
    setLoadingSessionId(null);
  }, [pathname]);

  useEffect(() => {
    if (!sessionId || !completedSessionIDs.has(sessionId)) return;
    clearCompletedSession(sessionId);
    const remaining = completedSessionIDs.size - 1;
    document.title =
      remaining > 0 ? `(${remaining}) Otto is ready - AutoGPT` : "AutoGPT";
  }, [sessionId, completedSessionIDs, clearCompletedSession]);

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  function handleSelectSession(id: string) {
    if (!isCopilotPage) {
      setLoadingSessionId(id);
      router.push(`/copilot?sessionId=${id}`);
    } else {
      setSessionId(id);
    }
  }

  function handleRenameClick(
    e: React.MouseEvent,
    id: string,
    title: string | null | undefined,
  ) {
    e.stopPropagation();
    setEditingSessionId(id);
    setEditingTitle(title || "");
  }

  function handleRenameSubmit(id: string) {
    const trimmed = editingTitle.trim();
    if (trimmed) {
      renameSession({ sessionId: id, data: { title: trimmed } });
    } else {
      setEditingSessionId(null);
    }
  }

  function handleDeleteClick(
    e: React.MouseEvent,
    id: string,
    title: string | null | undefined,
  ) {
    e.stopPropagation();
    if (isDeleting) return;
    setSessionToDelete({ id, title });
  }

  function handleConfirmDelete() {
    if (sessionToDelete) {
      deleteSession({ sessionId: sessionToDelete.id });
    }
  }

  function handleCancelDelete() {
    if (!isDeleting) {
      setSessionToDelete(null);
    }
  }

  function getDateGroup(dateString: string) {
    const date = new Date(dateString);
    const now = new Date();

    const startOfToday = new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate(),
    );
    const startOfDate = new Date(
      date.getFullYear(),
      date.getMonth(),
      date.getDate(),
    );
    const diffDays = Math.round(
      (startOfToday.getTime() - startOfDate.getTime()) / (1000 * 60 * 60 * 24),
    );

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return "Previous 7 days";
    if (diffDays < 30) return "Previous 30 days";

    const month = date.toLocaleDateString("en-US", { month: "long" });
    const year = date.getFullYear();
    return `${month} ${year}`;
  }

  function groupSessions(
    items: typeof sessions,
  ): { label: string; items: typeof sessions }[] {
    const groups: Map<string, typeof sessions> = new Map();
    for (const session of items) {
      const label = getDateGroup(session.updated_at);
      const existing = groups.get(label);
      if (existing) {
        existing.push(session);
      } else {
        groups.set(label, [session]);
      }
    }
    return Array.from(groups, ([label, items]) => ({ label, items }));
  }

  return (
    <>
      <div className="flex flex-col gap-5">
        {isLoadingSessions ? (
          <div className="flex min-h-[30rem] items-center justify-center py-4">
            <LoadingSpinner size="medium" className="text-neutral-600" />
          </div>
        ) : isSessionsError ? (
          <div className="px-3 py-4">
            <ErrorCard context="chat sessions" />
          </div>
        ) : sessions.length === 0 ? (
          <p className="py-4 text-center text-sm text-neutral-500">
            No conversations yet
          </p>
        ) : (
          groupSessions(sessions).map((group) => (
            <div key={group.label} className="flex flex-col">
              <span className="px-3 pb-0.5 text-xs font-medium text-zinc-600">
                {group.label}
              </span>
              {group.items.map((session) => (
                <div
                  key={session.id}
                  className={cn(
                    "relative w-full rounded-xl transition-colors",
                    session.id === activeSessionId
                      ? "bg-zinc-200/60"
                      : "hover:bg-sidebar-accent",
                  )}
                  onMouseEnter={() => setHoveredSessionId(session.id)}
                  onMouseLeave={() => setHoveredSessionId(null)}
                >
                  <button
                    onClick={() => handleSelectSession(session.id)}
                    className="w-full px-3 py-2.5 pr-10 text-left"
                  >
                    <div className="flex min-w-0 max-w-full items-center gap-2">
                      <div className="min-w-0 flex-1">
                        <Text
                          variant="body"
                          className={cn(
                            "truncate text-sm font-normal",
                            session.id === activeSessionId
                              ? "text-zinc-900"
                              : "text-zinc-900",
                          )}
                        >
                          <AnimatePresence mode="wait" initial={false}>
                            <motion.span
                              key={session.title || "untitled"}
                              initial={{ opacity: 0, y: 4 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -4 }}
                              transition={{ duration: 0.2 }}
                              className="block truncate"
                            >
                              {session.title || "Untitled chat"}
                            </motion.span>
                          </AnimatePresence>
                        </Text>
                      </div>
                      {loadingSessionId === session.id && (
                        <CircleNotch
                          className="h-4 w-4 shrink-0 animate-spin text-zinc-600"
                          weight="bold"
                        />
                      )}
                      {!loadingSessionId &&
                        session.is_processing &&
                        session.id !== activeSessionId &&
                        !completedSessionIDs.has(session.id) && (
                          <CircleNotch
                            className="h-4 w-4 shrink-0 animate-spin text-zinc-400"
                            weight="bold"
                          />
                        )}
                      {!loadingSessionId &&
                        completedSessionIDs.has(session.id) &&
                        session.id !== activeSessionId && (
                          <CheckCircle
                            className="h-4 w-4 shrink-0 text-green-500"
                            weight="fill"
                          />
                        )}
                    </div>
                  </button>
                  <AnimatePresence>
                    {hoveredSessionId === session.id && (
                      <motion.div
                        initial={{ x: "100%" }}
                        animate={{ x: 0 }}
                        exit={{ x: "100%" }}
                        transition={{
                          duration: 0.25,
                          ease: [0.32, 0.72, 0, 1],
                        }}
                        className="absolute right-0 top-0 flex h-full items-center"
                      >
                        <div
                          className="pointer-events-none h-full w-8 bg-gradient-to-r from-transparent"
                          style={{
                            ["--tw-gradient-to" as string]:
                              session.id === activeSessionId
                                ? "rgb(235 235 238)"
                                : "hsl(var(--sidebar-accent))",
                          }}
                        />
                        <div
                          className="flex h-full items-center gap-0.5 rounded-r-xl pr-2"
                          style={{
                            backgroundColor:
                              session.id === activeSessionId
                                ? "rgb(235 235 238)"
                                : "hsl(var(--sidebar-accent))",
                          }}
                        >
                          <button
                            onClick={(e) =>
                              handleRenameClick(e, session.id, session.title)
                            }
                            className="flex size-7 items-center justify-center rounded-xl transition-colors hover:bg-zinc-200"
                            aria-label="Rename chat"
                          >
                            <PencilSimple className="!size-[18px]" />
                          </button>
                          <button
                            onClick={(e) =>
                              handleDeleteClick(e, session.id, session.title)
                            }
                            disabled={isDeleting}
                            className="flex size-7 items-center justify-center rounded-xl transition-colors hover:bg-red-100 hover:text-red-600"
                            aria-label="Delete chat"
                          >
                            <Trash className="!size-[18px]" />
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {!isMobile && (
        <DeleteChatDialog
          session={sessionToDelete}
          isDeleting={isDeleting}
          onConfirm={handleConfirmDelete}
          onCancel={handleCancelDelete}
        />
      )}

      <Dialog
        title="Edit title"
        styling={{ maxWidth: "28rem", minWidth: "auto" }}
        controlled={{
          isOpen: !!editingSessionId,
          set: async (open) => {
            if (!open) setEditingSessionId(null);
          },
        }}
      >
        <Dialog.Content>
          <p className="text-sm text-zinc-500">Please enter a new title</p>
          <div className="relative mt-3">
            <input
              ref={renameInputRef}
              type="text"
              value={editingTitle}
              onChange={(e) => setEditingTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && editingSessionId) {
                  handleRenameSubmit(editingSessionId);
                }
              }}
              className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2.5 pr-10 text-sm text-zinc-900 outline-none focus:border-zinc-300 focus:ring-1 focus:ring-zinc-300"
            />
            {editingTitle && (
              <button
                onClick={() => setEditingTitle("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-600"
              >
                <X className="size-4" weight="bold" />
              </button>
            )}
          </div>
          <Dialog.Footer>
            <Button
              variant="secondary"
              size="medium"
              onClick={() => setEditingSessionId(null)}
              disabled={isRenaming}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              size="medium"
              onClick={() => {
                if (editingSessionId) handleRenameSubmit(editingSessionId);
              }}
              disabled={!editingTitle.trim() || isRenaming}
              loading={isRenaming}
            >
              Confirm
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
