/**
 * CoPilot Bot — Core logic using Vercel Chat SDK.
 *
 * Server-level ownership model:
 * - The first person to authenticate a server becomes its "owner".
 * - All users in the server get their own CoPilot sessions, all billed
 *   to the owner's AutoGPT account and visible in their platform account.
 * - If a server is not linked, the triggering user is DM'd a setup link.
 */

import { Chat } from "chat";
import type { Adapter, StateAdapter, Thread, Message } from "chat";
import { PlatformAPI, PlatformAPIError } from "./platform-api.js";
import type { Config } from "./config.js";

/** Thread state persisted across messages in a conversation. */
export interface BotThreadState {
  /** CoPilot session ID for this specific user×server conversation. */
  sessionId?: string;
  /** Pending setup token (sent while waiting for owner to link). */
  pendingLinkToken?: string;
}

type BotThread = Thread<BotThreadState>;

export async function createBot(config: Config, stateAdapter: StateAdapter) {
  const api = new PlatformAPI(config.autogptApiUrl);

  const adapters: Record<string, Adapter> = {};

  if (config.discord) {
    const { createDiscordAdapter } = await import("@chat-adapter/discord");
    adapters.discord = createDiscordAdapter();
  }

  if (config.telegram) {
    const { createTelegramAdapter } = await import("@chat-adapter/telegram");
    adapters.telegram = createTelegramAdapter();
  }

  if (config.slack) {
    const { createSlackAdapter } = await import("@chat-adapter/slack");
    adapters.slack = createSlackAdapter();
  }

  if (Object.keys(adapters).length === 0) {
    throw new Error(
      "No adapters configured. Set at least one of: " +
        "DISCORD_BOT_TOKEN, TELEGRAM_BOT_TOKEN, SLACK_BOT_TOKEN",
    );
  }

  const bot = new Chat<typeof adapters, BotThreadState>({
    userName: "copilot",
    adapters,
    state: stateAdapter,
  });

  // ── New mention (first message in a thread) ──────────────────────────

  bot.onNewMention(async (thread, message) => {
    const platform = getPlatformName(thread.id);
    const serverId = getServerId(thread.id);
    const platformUserId = message.author.userId;

    console.log(
      `[bot] Mention in ${platform} server ${serverId} from user ${platformUserId}`,
    );

    if (isHelpCommand(message.text)) {
      await thread.post(helpText());
      return;
    }

    // Pass userId so DM fallback works: already-linked users aren't re-prompted
    const resolved = await api.resolve(platform, serverId, platformUserId);

    if (!resolved.linked) {
      await handleUnlinkedServer(thread, message, platform, serverId, api, bot);
      return;
    }

    await thread.subscribe();
    await handleCoPilotMessage(
      thread, message.text, platform, serverId, platformUserId, api,
    );
  });

  // ── Follow-up messages in a subscribed thread ────────────────────────

  bot.onSubscribedMessage(async (thread, message) => {
    const platform = getPlatformName(thread.id);
    const serverId = getServerId(thread.id);
    const platformUserId = message.author.userId;

    if (isHelpCommand(message.text)) {
      await thread.post(helpText());
      return;
    }

    // Re-check linking in case the owner just completed setup
    const resolved = await api.resolve(platform, serverId, platformUserId);

    if (!resolved.linked) {
      await handleUnlinkedServer(thread, message, platform, serverId, api, bot);
      return;
    }

    await handleCoPilotMessage(
      thread, message.text, platform, serverId, platformUserId, api,
    );
  });

  return bot;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Extract the adapter/platform name from a thread ID.
 * Thread ID format: "adapter:channelOrServerId:threadId"
 */
function getPlatformName(threadId: string): string {
  return threadId.split(":")[0] ?? "unknown";
}

/**
 * Extract the server/channel identifier from a thread ID.
 * Used as platform_server_id for all API calls.
 */
function getServerId(threadId: string): string {
  return threadId.split(":")[1] ?? threadId;
}

function isHelpCommand(text: string): boolean {
  return text.trim().toLowerCase().startsWith("/help");
}

/**
 * Whether this is a direct message (no "server" context).
 * For Telegram: DM chat IDs are positive and match the user's ID.
 * For Discord: DMs use a guild ID of "@me".
 */
function isDM(platform: string, serverId: string, platformUserId: string): boolean {
  if (platform === "telegram") return !serverId.startsWith("-");
  if (platform === "discord") return serverId === "@me";
  return serverId === platformUserId;
}

/**
 * Build context-aware copy for the setup message and confirm button.
 * Returns strings appropriate for whether this is a personal DM link
 * or a group/server link.
 */
function getLinkContext(
  platform: string,
  serverId: string,
  platformUserId: string,
): { isDirect: boolean; contextLabel: string; serverName: string | undefined } {
  const PLATFORM_DISPLAY: Record<string, string> = {
    discord: "Discord",
    telegram: "Telegram",
    slack: "Slack",
  };
  const display = PLATFORM_DISPLAY[platform] ?? platform;

  if (isDM(platform, serverId, platformUserId)) {
    return {
      isDirect: true,
      contextLabel: `your ${display} account`,
      serverName: undefined,
    };
  }

  return {
    isDirect: false,
    contextLabel: `this ${display} group`,
    serverName: undefined,
  };
}

/**
 * Attempt to DM a user via bot.openDM().
 * Returns true if the DM was sent, false if it failed (e.g. Telegram users
 * who haven't started a private chat with the bot, or unknown user ID format).
 */
async function tryDM(
  bot: Chat<Record<string, Adapter>, BotThreadState>,
  message: Message,
  text: string,
): Promise<boolean> {
  try {
    const dmThread = await bot.openDM(message.author);
    await dmThread.post(text);
    return true;
  } catch (err) {
    console.log(
      `[bot] DM unavailable for ${message.author.userId}, posting in thread instead:`,
      err instanceof Error ? err.message : String(err),
    );
    return false;
  }
}

/**
 * Handle a message in an unlinked server.
 * DMs the triggering user a one-time setup link — never posts in the channel.
 */
async function handleUnlinkedServer(
  thread: BotThread,
  message: Message,
  platform: string,
  serverId: string,
  api: PlatformAPI,
  bot: Chat<Record<string, Adapter>, BotThreadState>,
) {
  const { isDirect, contextLabel } = getLinkContext(
    platform,
    serverId,
    message.author.userId,
  );

  console.log(
    `[bot] ${isDirect ? "DM" : "Group"} ${platform}:${serverId} not linked, sending setup link`,
  );

  try {
    const linkResult = await api.createLinkToken({
      platform,
      platformServerId: serverId,
      platformUserId: message.author.userId,
      platformUsername: message.author.fullName ?? message.author.userName,
    });

    const setupMessage = isDirect
      ? `To use AutoPilot, link ${contextLabel} to AutoGPT:\n\n${linkResult.link_url}\n\nThis link expires in 30 minutes.`
      : `To set up AutoPilot for ${contextLabel}, connect your AutoGPT account.\n\nOnce linked, everyone here can use AutoPilot — usage appears in your AutoGPT account.\n\n${linkResult.link_url}\n\nThis link expires in 30 minutes.`;

    if (isDirect) {
      // Already in a DM — post the link directly in this conversation
      await thread.post(setupMessage);
    } else {
      // In a group — try to DM so the link isn't public, fall back to group post
      const dmSent = await tryDM(bot, message, setupMessage);
      if (dmSent) {
        await thread.post(
          `I've sent you a DM with a setup link! Once connected, everyone here can chat with AutoPilot.`,
        );
      } else {
        await thread.post(setupMessage);
      }
    }

    await thread.setState({ pendingLinkToken: linkResult.token });
  } catch (err) {
    if (err instanceof PlatformAPIError && err.status === 409) {
      // Race condition: server was linked between resolve and createLinkToken
      const resolved = await api.resolve(platform, serverId);
      if (resolved.linked) {
        await thread.subscribe();
        await handleCoPilotMessage(
          thread, message.text, platform, serverId, message.author.userId, api,
        );
        return;
      }
    }

    console.error("[bot] Failed to create link token:", err);
    await thread.post(
      "Sorry, I couldn't set up account linking right now. Please try again later.",
    );
  }
}

/**
 * Forward a message to CoPilot and post the response.
 * Each (server, platform_user) pair gets its own session under the owner's account.
 */
async function handleCoPilotMessage(
  thread: BotThread,
  text: string,
  platform: string,
  serverId: string,
  platformUserId: string,
  api: PlatformAPI,
) {
  const state = await thread.state;
  let sessionId = state?.sessionId;

  await thread.startTyping();

  try {
    if (!sessionId) {
      sessionId = await api.createChatSession(platform, serverId, platformUserId);
      await thread.setState({ ...state, sessionId });
      console.log(
        `[bot] Created session ${sessionId} for ${platform}:${serverId}:${platformUserId}`,
      );
    }

    // Collect the full response before posting to avoid "empty message" errors
    const stream = api.streamChat(
      platform, serverId, platformUserId, text, sessionId,
    );
    let response = "";
    for await (const chunk of stream) {
      response += chunk;
    }

    if (response.trim()) {
      await thread.post(response);
    } else {
      await thread.post(
        "I processed your message but didn't generate a response. Please try again.",
      );
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(
      `[bot] CoPilot error for ${platform}:${serverId}:${platformUserId}:`, msg,
    );
    await thread.post(
      "Sorry, I ran into an issue processing your message. Please try again.",
    );
  }
}

function helpText(): string {
  return (
    `**AutoPilot** — Your AutoGPT assistant\n\n` +
    `**Getting started:**\n` +
    `• The first person to mention me will receive a DM with a one-time setup link\n` +
    `• Once set up, everyone in the server can chat with AutoPilot\n` +
    `• Each person gets their own private conversation\n` +
    `• All conversations appear in the setup owner's AutoGPT account\n\n` +
    `**Commands:**\n` +
    `• \`/help\` — Show this message\n\n` +
    `Powered by [AutoGPT](https://platform.agpt.co)`
  );
}
