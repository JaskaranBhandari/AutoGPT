/**
 * Slack webhook handler.
 * Vercel route: POST /api/webhooks/slack
 *
 * Configure in your Slack App settings under "Event Subscriptions":
 *   Request URL: https://your-domain.com/api/webhooks/slack
 */

import { getBotInstance } from "../_bot.js";

export async function POST(request: Request): Promise<Response> {
  const bot = await getBotInstance();
  return bot.webhooks.slack(request);
}
