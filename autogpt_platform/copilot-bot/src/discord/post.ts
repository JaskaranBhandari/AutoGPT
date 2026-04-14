/**
 * Direct Discord REST API posting — used when the Chat SDK's Card/Actions
 * abstraction wraps our content in an embed with duplicate fallback text.
 *
 * Posts `content` + `components` inline so the text renders as a normal
 * message body with buttons underneath, no embed clutter.
 */

import type { Component } from "./types.js";

const DISCORD_API = "https://discord.com/api/v10";

/**
 * Extract the Discord channel/thread ID from a Chat SDK thread ID.
 * Thread ID format: "discord:{guildId}:{channelId}[:{threadId}]"
 * Posts to the deepest segment — so in a Discord thread we post to the thread,
 * not the parent channel.
 *
 * Throws if the thread ID doesn't start with "discord:" — a format change in
 * the Chat SDK would otherwise silently route messages to the wrong channel.
 */
export function discordPostTarget(threadId: string): string {
  const parts = threadId.split(":");
  if (parts[0] !== "discord" || parts.length < 3) {
    throw new Error(
      `Unexpected Discord thread ID format: "${threadId}" (expected "discord:guild:channel[:thread]")`,
    );
  }
  return parts[3] ?? parts[2];
}

/**
 * Post a message to a Discord channel with content + interactive components.
 * Returns the message ID.
 */
export async function postDiscordMessage(
  channelId: string,
  content: string,
  components: Component[],
): Promise<string | null> {
  const token = process.env.DISCORD_BOT_TOKEN;
  if (!token) {
    throw new Error("DISCORD_BOT_TOKEN not set");
  }

  const res = await fetch(`${DISCORD_API}/channels/${channelId}/messages`, {
    method: "POST",
    headers: {
      Authorization: `Bot ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ content, components }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Discord post failed (${res.status}): ${text}`);
  }

  const data = (await res.json()) as { id: string };
  return data.id;
}
