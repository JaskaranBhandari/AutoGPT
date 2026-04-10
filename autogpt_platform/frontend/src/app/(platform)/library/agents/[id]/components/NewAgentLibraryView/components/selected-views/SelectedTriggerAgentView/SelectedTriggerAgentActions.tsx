"use client";

import {
  getGetV2ListTriggerAgentsQueryKey,
  useDeleteV2DeleteLibraryAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  ArrowSquareOutIcon,
  PencilSimpleIcon,
  TrashIcon,
} from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

interface Props {
  parentAgent: LibraryAgent;
  triggerAgent: LibraryAgent;
  onDeleted?: () => void;
}

export function SelectedTriggerAgentActions({
  parentAgent,
  triggerAgent,
  onDeleted,
}: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const deleteMutation = useDeleteV2DeleteLibraryAgent({
    mutation: {
      onSuccess: async () => {
        toast({ title: "Trigger removed" });
        queryClient.invalidateQueries({
          queryKey: getGetV2ListTriggerAgentsQueryKey(parentAgent.id),
        });
        setShowDeleteDialog(false);
        onDeleted?.();
      },
      onError: (error) => {
        toast({
          title: "Failed to remove trigger",
          description:
            error instanceof Error
              ? error.message
              : "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  function handleDelete() {
    deleteMutation.mutate({ libraryAgentId: triggerAgent.id });
  }

  return (
    <>
      <div className="my-4 flex flex-col items-center gap-3">
        <Link
          href={`/library/agents/${triggerAgent.id}`}
          aria-label="View in library"
        >
          <Button variant="icon" size="icon" aria-label="View in library">
            <ArrowSquareOutIcon
              weight="bold"
              size={18}
              className="text-zinc-700"
            />
          </Button>
        </Link>
        <Link
          href={`/build?flowID=${triggerAgent.graph_id}`}
          aria-label="Open in builder"
        >
          <Button variant="icon" size="icon" aria-label="Open in builder">
            <PencilSimpleIcon
              weight="bold"
              size={18}
              className="text-zinc-700"
            />
          </Button>
        </Link>
        <Button
          variant="icon"
          size="icon"
          aria-label="Remove trigger"
          onClick={() => setShowDeleteDialog(true)}
          disabled={deleteMutation.isPending}
        >
          {deleteMutation.isPending ? (
            <LoadingSpinner size="small" />
          ) : (
            <TrashIcon weight="bold" size={18} />
          )}
        </Button>
      </div>

      <Dialog
        controlled={{
          isOpen: showDeleteDialog,
          set: setShowDeleteDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Remove trigger"
      >
        <Dialog.Content>
          <Text variant="large">
            Are you sure you want to remove this trigger? The trigger agent and
            its schedule will be deleted. This action cannot be undone.
          </Text>
          <Dialog.Footer>
            <Button
              variant="secondary"
              onClick={() => setShowDeleteDialog(false)}
              disabled={deleteMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              loading={deleteMutation.isPending}
            >
              Remove trigger
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
