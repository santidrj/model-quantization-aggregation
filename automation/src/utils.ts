import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

export async function promptLine(question: string): Promise<string> {
  const rl = readline.createInterface({ input, output });
  try {
    return (await rl.question(question)).trim();
  } finally {
    rl.close();
  }
}

export function normalizeLabel(label: string): string {
  return label
    .trim()
    .toLowerCase()
    .replace(/[’']/g, "'")
    .replace(/\s+/g, " ");
}

/** Strip trailing " (1)" / " (2)" origin markers from a rendered label. */
export function stripOriginSuffix(text: string): string {
  return text.replace(/\s*\([12]\)\s*$/, "").trim();
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function isLoginUrl(url: string): boolean {
  return /\/user\/login/i.test(url);
}
