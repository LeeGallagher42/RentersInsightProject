'use strict';
const assert = require('node:assert/strict');
const { SyntheticDownstream, MemoryStore, brokenWorkflow, hardenedWorkflow } = require('./demo');

async function run() {
  const results = [];
  async function test(name, fn) {
    await fn();
    results.push({ name, result: 'PASS' });
  }

  await test('Broken workflow duplicates side effect', async () => {
    const d = new SyntheticDownstream();
    const e = { id: 'evt-dup', payload: { order: 42 } };
    await brokenWorkflow(e, d);
    await brokenWorkflow(e, d);
    assert.equal(d.sideEffects.length, 2);
  });

  await test('Hardened workflow skips duplicate', async () => {
    const d = new SyntheticDownstream();
    const s = new MemoryStore();
    const e = { id: 'evt-dup', payload: { order: 42 } };
    await hardenedWorkflow(e, d, s);
    const second = await hardenedWorkflow(e, d, s);
    assert.equal(d.sideEffects.length, 1);
    assert.equal(second.duplicateSkipped, true);
    assert.equal(s.logs.some(x => x.event === 'duplicate_skipped'), true);
  });

  await test('Transient failures recover with bounded backoff', async () => {
    const d = new SyntheticDownstream({ 'evt-retry': ['timeout', 'timeout', 'ok'] });
    const s = new MemoryStore();
    const out = await hardenedWorkflow({ id: 'evt-retry', payload: {} }, d, s, { baseDelayMs: 1 });
    assert.equal(out.accepted, true);
    assert.equal(d.attempts.get('evt-retry'), 3);
    assert.equal(s.get('evt-retry').status, 'completed');
  });

  await test('Permanent failure fails safe and alerts', async () => {
    const d = new SyntheticDownstream({ 'evt-fail': ['reject'] });
    const s = new MemoryStore();
    const out = await hardenedWorkflow({ id: 'evt-fail', payload: {} }, d, s, { baseDelayMs: 1 });
    assert.equal(out.failedSafe, true);
    assert.equal(s.get('evt-fail').status, 'failed');
    assert.equal(s.alerts.length, 1);
    assert.equal(s.alerts[0].severity, 'high');
  });

  await test('Structured audit log records recovery path', async () => {
    const d = new SyntheticDownstream({ 'evt-log': ['timeout', 'ok'] });
    const s = new MemoryStore();
    await hardenedWorkflow({ id: 'evt-log', payload: {} }, d, s, { baseDelayMs: 1 });
    assert.deepEqual(s.logs.map(x => x.event), ['attempt_failed', 'completed']);
  });

  console.log(JSON.stringify({ passed: results.length, failed: 0, results }, null, 2));
}

run().catch(err => { console.error(err); process.exit(1); });
