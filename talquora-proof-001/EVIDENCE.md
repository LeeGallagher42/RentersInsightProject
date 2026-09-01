# Proof 001 — Synthetic Evidence Record

This file records the reproducible demo evidence behind the buyer-facing README. No client or production claims are made.

## Local automated test run

Command: `node test.js`

Result: **5 passed, 0 failed**.

Validated behaviors:

1. Broken workflow duplicates an external side effect on duplicate delivery.
2. Hardened workflow suppresses the duplicate and logs `duplicate_skipped`.
3. Two transient synthetic timeouts recover on attempt 3.
4. A permanent synthetic rejection creates durable failed state plus a high-severity alert.
5. Structured logs preserve the retry/recovery path.

## Make seed evidence

A successful synthetic seed run wrote the following evidence structure to the Proof 001 workbook:

### processed_events
- `evt-dup` — completed, attempt 1, accepted.
- `evt-retry` — completed, attempt 3, accepted after two transient timeouts.
- `evt-fail` — failed, attempt 1, permanent rejection safely recorded.

### logs
- completion for `evt-dup`;
- `duplicate_skipped` for repeated `evt-dup`;
- two `attempt_failed` entries then completion for `evt-retry`;
- `attempt_failed` then `failed_safe` for `evt-fail`.

### alerts
- high-severity `workflow_failure` for `evt-fail`.

### tests
Five synthetic tests were recorded as PASS, matching the buyer-facing behavior categories.

## Scope / honesty boundary

This is a synthetic reliability demonstration. It does not represent customer infrastructure, customer data, realised revenue, uptime improvement, incident reduction, or production SLAs.
