'use strict';

class SyntheticDownstream {
  constructor(plan = {}) {
    this.plan = new Map(Object.entries(plan));
    this.sideEffects = [];
    this.attempts = new Map();
  }

  async send(event) {
    const n = (this.attempts.get(event.id) || 0) + 1;
    this.attempts.set(event.id, n);
    const behavior = this.plan.get(event.id) || [];
    const outcome = behavior[n - 1] || 'ok';
    if (outcome === 'timeout') throw Object.assign(new Error(`temporary timeout ${n}`), { transient: true });
    if (outcome === 'reject') throw Object.assign(new Error('permanent downstream rejection'), { transient: false });
    this.sideEffects.push({ eventId: event.id, payload: event.payload, attempt: n });
    return { accepted: true, attempt: n };
  }
}

async function brokenWorkflow(event, downstream) {
  return downstream.send(event);
}

class MemoryStore {
  constructor() {
    this.records = new Map();
    this.logs = [];
    this.alerts = [];
  }
  get(id) { return this.records.get(id); }
  put(id, value) { this.records.set(id, value); }
  log(entry) { this.logs.push(entry); }
  alert(entry) { this.alerts.push(entry); }
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function hardenedWorkflow(event, downstream, store, options = {}) {
  const maxAttempts = options.maxAttempts || 3;
  const baseDelayMs = options.baseDelayMs ?? 5;

  const existing = store.get(event.id);
  if (existing?.status === 'completed') {
    store.log({ eventId: event.id, level: 'info', event: 'duplicate_skipped', message: 'Prior completed result returned; no second side effect.' });
    return { ...existing.result, duplicateSkipped: true };
  }

  store.put(event.id, { status: 'processing', attempts: 0 });

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const result = await downstream.send(event);
      store.put(event.id, { status: 'completed', attempts: attempt, result });
      store.log({ eventId: event.id, level: 'info', event: 'completed', attempt, message: 'Side effect completed.' });
      return result;
    } catch (err) {
      store.log({ eventId: event.id, level: 'error', event: 'attempt_failed', attempt, message: err.message });
      const retryable = err.transient === true && attempt < maxAttempts;
      if (retryable) {
        await sleep(baseDelayMs * (2 ** (attempt - 1)));
        continue;
      }

      store.put(event.id, { status: 'failed', attempts: attempt, error: err.message });
      store.log({ eventId: event.id, level: 'error', event: 'failed_safe', attempt, message: 'Failure recorded; alert raised.' });
      store.alert({ severity: 'high', type: 'workflow_failure', eventId: event.id, message: err.message });
      return { accepted: false, failedSafe: true, error: err.message };
    }
  }
}

module.exports = { SyntheticDownstream, MemoryStore, brokenWorkflow, hardenedWorkflow };
