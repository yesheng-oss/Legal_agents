import assert from "node:assert/strict";
import { spawn } from "node:child_process";

const BRIDGE_SCRIPT = "C:/Users/22234/Desktop/gbrain_query_bridge_5005.mjs";
const SOURCE_DIR = "C:/Users/22234/Desktop/gb";
const PORT = 5015;
const BASE_URL = `http://127.0.0.1:${PORT}`;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForHealth() {
  for (let i = 0; i < 30; i += 1) {
    try {
      const response = await fetch(`${BASE_URL}/health`);
      if (response.ok) {
        return;
      }
    } catch {}
    await sleep(500);
  }
  throw new Error("bridge_not_ready");
}

async function main() {
  const child = spawn(
    "node",
    [BRIDGE_SCRIPT],
    {
      env: {
        ...process.env,
        GBRAIN_QUERY_BRIDGE_PORT: String(PORT),
        GBRAIN_SOURCE_DIR: SOURCE_DIR,
      },
      stdio: "ignore",
      windowsHide: true,
    },
  );

  try {
    await waitForHealth();

    const listResponse = await fetch(`${BASE_URL}/list`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit: 5 }),
    });
    assert.equal(listResponse.status, 200, "POST /list should succeed");
    const listPayload = await listResponse.json();
    assert.equal(listPayload.ok, true);
    assert.equal(listPayload.source_dir, SOURCE_DIR);
    assert.ok(Array.isArray(listPayload.items));
    assert.ok(listPayload.items.length > 0);
    assert.ok(listPayload.items[0].slug, "list item should contain slug");

    const getResponse = await fetch(`${BASE_URL}/get`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug: listPayload.items[0].slug }),
    });
    assert.equal(getResponse.status, 200, "POST /get should succeed");
    const getPayload = await getResponse.json();
    assert.equal(getPayload.ok, true);
    assert.equal(getPayload.slug, listPayload.items[0].slug);
    assert.equal(typeof getPayload.content, "string");
    assert.ok(getPayload.content.length > 0);
  } finally {
    child.kill();
  }
}

await main();
