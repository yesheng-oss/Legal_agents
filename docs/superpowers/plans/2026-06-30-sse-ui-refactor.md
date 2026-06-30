# SSE UI Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make chat streaming a stable SSE contract and refresh the built-in legal workbench interface.

**Architecture:** Keep the existing FastAPI app and embedded `DEMO_HTML` structure. Add focused backend helpers for SSE formatting and response headers, then update the browser UI to consume streamed events and present a cleaner three-panel workbench.

**Tech Stack:** FastAPI, StreamingResponse, vanilla HTML/CSS/JavaScript, pytest, FastAPI TestClient.

---

### Task 1: SSE Contract

**Files:**
- Modify: `src/api.py`
- Test: `tests/test_api.py`

- [ ] Add a test that `/chat/stream` returns `text/event-stream`, disables buffering/caching, and emits ordered `meta`, `delta`, `references`, `memory`, `steps`, and `done` events.
- [ ] Run the focused test and verify it fails before implementation.
- [ ] Extract SSE formatting to a small helper and return `StreamingResponse` with stable headers.
- [ ] Run the focused test and verify it passes.

### Task 2: Frontend Workbench

**Files:**
- Modify: `src/api.py`
- Test: `tests/test_api.py`

- [ ] Add or update the `/docs` HTML test so it checks for the refreshed workbench layout, SSE status copy, and stream parser functions.
- [ ] Run the focused test and verify it fails before implementation.
- [ ] Replace the embedded `DEMO_HTML` with a cleaner three-panel layout and a fetch-based SSE parser for `POST /chat/stream`.
- [ ] Run the focused test and verify it passes.

### Task 3: Verification

**Files:**
- Test: `tests/test_api.py`

- [ ] Run `python -m pytest tests/test_api.py -q`.
- [ ] Check that the output is passing and report any residual risk.
