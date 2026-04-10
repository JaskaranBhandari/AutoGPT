"use client";

import {
  getGetV2ListTriggerAgentsQueryKey,
  useDeleteV2DeleteLibraryAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { DotsThreeVerticalIcon } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

interface Props {
  parentAgent: LibraryAgent;
  triggerAgent: LibraryAgent;
  onDeleted?: () => void;
}

export function TriggerAgentActionsDropdown({
  parentAgent,
  triggerAgent,
  onDeleted,
}: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const { mutateAsync: deleteLibraryAgent, isPending: isDeleting } =
    useDeleteV2DeleteLibraryAgent();

  async function handleDelete() {
    try {
      await deleteLibraryAgent({ libraryAgentId: triggerAgent.id });

      toast({ title: "Trigger removed" });

      queryClient.invalidateQueries({
        queryKey: getGetV2ListTriggerAgentsQueryKey(parentAgent.id),
      });

      setShowDeleteDialog(false);
      onDeleted?.();
    } catch (error: unknown) {
      toast({
        title: "Failed to remove trigger",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            className="ml-auto shrink-0 rounded p-1 hover:bg-gray-100"
            onClick={(e) => e.stopPropagation()}
            aria-label="More actions"
          >
            <DotsThreeVerticalIcon className="h-5 w-5 text-gray-400" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem asChild>
            <Link
              href={`/library/agents/${triggerAgent.id}`}
              onClick={(e) => e.stopPropagation()}
            >
              View in library
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link
              href={`/build?flowID=${triggerAgent.graph_id}`}
              onClick={(e) => e.stopPropagation()}
            >
              Open in builder
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              setShowDeleteDialog(true);
            }}
          >
            Remove trigger
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog
        controlled={{
          isOpen: showDeleteDialog,
          set: setShowDeleteDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Remove trigger"
      >
        <Dialog.Content>
          <div>
            <Text variant="large">
              Are you sure you want to remove this trigger? The trigger agent
              and its schedule will be deleted. This action cannot be undone.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeleting}
                onClick={() => setShowDeleteDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                loading={isDeleting}
              >
                Remove trigger
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
