/**
 * Telegram webhook handler.
 * Vercel route: POST /api/webhooks/telegram
 *
 * In production (Vercel), register this URL with Telegram:
 *   curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
 *     -H "Content-Type: application/json" \
 *     -d '{"url": "https://your-domain.com/api/webhooks/telegram", "secret_token": "$TELEGRAM_WEBHOOK_SECRET_TOKEN"}'
 *
 * In local dev, the adapter uses polling automatically — no webhook needed.
 */

import { getBotInstance } from "../_bot.js";

export async function POST(request: Request): Promise<Response> {
  const bot = await getBotInstance();
  return bot.webhooks.telegram(request);
}
