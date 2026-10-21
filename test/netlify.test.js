import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { netlifyHandler, storeNameFor } from '../lib/netlify.js';
import { CONTENT_SECURITY_POLICY } from '../server.js';

// In-memory stand-in for a Netlify Blobs store, including ETags and conditional writes.
// Every call yields to the event loop first, so concurrent requests really interleave.
function fakeBlobs() {
  const blobs = new Map();
  let version = 0;
  const tick = () => new Promise((resolve) => setTimeout(resolve, Math.random() * 3));
  return {
    blobs,
    async get(key, { type } = {}) {
      await tick();
      if (!blobs.has(key)) return null;
      const { text } = blobs.get(key);
      return type === 'json' ? JSON.parse(text) : text;
    },
    async getWithMetadata(key, { type } = {}) {
      await tick();
      if (!blobs.has(key)) return null;
      const { text, etag } = blobs.get(key);
      return { data: type === 'json' ? JSON.parse(text) : text, etag, metadata: {} };
    },
    async setJSON(key, value, { onlyIfMatch, onlyIfNew } = {}) {
      await tick();
      const existing = blobs.get(key);
      if (onlyIfNew && existing) return { modified: false };
      if (onlyIfMatch && existing?.etag !== onlyIfMatch) return { modified: false };
      const etag = `"v${++version}"`;
      blobs.set(key, { text: JSON.stringify(value), etag });
      return { modified: true, etag };
    },
    async delete(key) {
      await tick();
      blobs.delete(key);
    },
  };
}

const ENV = { SEED_USERNAME: 'Tester', SEED_PASSWORD: 'correct horse battery' };
const env = (vars) => (key) => vars[key];
const production = { deploy: { context: 'production' } };

// A client for one function instance, holding its own cookie like a browser would.
function client(handler, ip = '203.0.113.7') {
  let cookie = '';
  return async (method, path, body) => {
    const headers = {};
    if (cookie) headers.cookie = cookie;
    if (body !== undefined) headers['content-type'] = 'application/json';
    const res = await handler(new Request(`https://lemma.example${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) }), { ip, ...production });
    const setCookie = res.headers.get('set-cookie');
    if (setCookie) cookie = setCookie.startsWith('sid=;') ? '' : setCookie.split(';')[0];
    const text = await res.text();
    return { status: res.status, body: text ? JSON.parse(text) : null, setCookie };
  };
}

test('creates the account from the environment on first use and signs in with a Secure cookie', async () => {
  const store = fakeBlobs();
  const call = client(netlifyHandler(() => store, env(ENV)));
  const login = await call('POST', '/api/login', { username: 'tester', password: ENV.SEED_PASSWORD });
  assert.equal(login.status, 200);
  assert.match(login.setCookie, /HttpOnly/);
  assert.match(login.setCookie, /Secure/);
  assert.deepEqual((await call('GET', '/api/me')).body, { username: 'tester' });
  assert.ok(![...store.blobs.values()].some((b) => b.text.includes(ENV.SEED_PASSWORD)), 'password stored only as a hash');
});

test('progress and reviews survive a cold start', async () => {
  const store = fakeBlobs();
  const first = client(netlifyHandler(() => store, env(ENV)));
  await first('POST', '/api/login', { username: 'tester', password: ENV.SEED_PASSWORD });
  await first('PUT', '/api/progress', { done: { 'mx-1': true }, xp: 15, streak: 1, last: '2026-10-21' });
  const state = { phase: 'review', step: 0, ivl: 4, ease: 2.5, reps: 1, lapses: 0, due: 2_000_000, last: 1_000_000, intro: 1_000_000 };
  assert.equal((await first('PUT', '/api/reviews', { cards: { 'mx-dot-def': state } })).status, 200);

  const second = client(netlifyHandler(() => store, env(ENV)));
  await second('POST', '/api/login', { username: 'tester', password: ENV.SEED_PASSWORD });
  assert.deepEqual((await second('GET', '/api/progress')).body.done, { 'mx-1': true });
  assert.deepEqual((await second('GET', '/api/reviews')).body.cards['mx-dot-def'], state);
  assert.ok(store.blobs.has('progress/tester') && store.blobs.has('reviews/tester'));
});

test('the seed never overwrites an existing account', async () => {
  const store = fakeBlobs();
  await client(netlifyHandler(() => store, env(ENV)))('GET', '/api/me');
  const changed = client(netlifyHandler(() => store, env({ ...ENV, SEED_PASSWORD: 'something else' })));
  assert.equal((await changed('POST', '/api/login', { username: 'tester', password: 'something else' })).status, 401);
  assert.equal((await changed('POST', '/api/login', { username: 'tester', password: ENV.SEED_PASSWORD })).status, 200);
});

test('without seed settings no account is created', async () => {
  const store = fakeBlobs();
  const call = client(netlifyHandler(() => store, env({})));
  assert.equal((await call('POST', '/api/login', { username: 'tester', password: 'anything' })).status, 401);
  assert.ok(![...store.blobs.keys()].some((k) => k.startsWith('users/')));
});

test('sign-in lockout holds across function instances', async () => {
  const store = fakeBlobs();
  const a = client(netlifyHandler(() => store, env(ENV)));
  for (let i = 0; i < 10; i++) assert.equal((await a('POST', '/api/login', { username: 'tester', password: `guess${i}` })).status, 401);
  const b = client(netlifyHandler(() => store, env(ENV)));
  assert.equal((await b('POST', '/api/login', { username: 'tester', password: ENV.SEED_PASSWORD })).status, 429);
  const otherIp = client(netlifyHandler(() => store, env(ENV)), '198.51.100.9');
  assert.equal((await otherIp('POST', '/api/login', { username: 'tester', password: ENV.SEED_PASSWORD })).status, 200);
});

test('unknown routes and unauthenticated requests are refused', async () => {
  const call = client(netlifyHandler(() => fakeBlobs(), env(ENV)));
  assert.equal((await call('GET', '/api/nope')).status, 404);
  assert.equal((await call('GET', '/api/progress')).status, 401);
  assert.equal((await call('PUT', '/api/reviews', { cards: {} })).status, 401);
});

test('netlify.toml publishes public/ with the same security policy as the Node server', async () => {
  const toml = await readFile(new URL('../netlify.toml', import.meta.url), 'utf8');
  assert.match(toml, /publish = "public"/);
  assert.equal(toml.match(/Content-Security-Policy = "([^"]+)"/)[1], CONTENT_SECURITY_POLICY);
  const fn = await import('../netlify/functions/api.mjs');
  assert.equal(typeof fn.default, 'function');
  assert.equal(fn.config.path, '/api/*');
});

test('simultaneous saves from two devices both land', async () => {
  const store = fakeBlobs();
  const phone = client(netlifyHandler(() => store, env(ENV)));
  const laptop = client(netlifyHandler(() => store, env(ENV)));
  await phone('POST', '/api/login', { username: 'tester', password: ENV.SEED_PASSWORD });
  await laptop('POST', '/api/login', { username: 'tester', password: ENV.SEED_PASSWORD });
  const state = (last) => ({ phase: 'review', step: 0, ivl: 1, ease: 2.5, reps: 1, lapses: 0, due: last + 1, last, intro: last });
  const ids = Array.from({ length: 12 }, (_, i) => `card-${i}`);
  await Promise.all(ids.map((id, i) => (i % 2 ? phone : laptop)('PUT', '/api/reviews', { cards: { [id]: state(1000 + i) } })));
  await Promise.all([
    phone('PUT', '/api/progress', { done: { 'mx-1': true }, xp: 10 }),
    laptop('PUT', '/api/progress', { done: { 'gen-1': true }, xp: 15 }),
  ]);
  const { cards } = (await phone('GET', '/api/reviews')).body;
  assert.deepEqual(Object.keys(cards).sort(), [...ids].sort());
  assert.deepEqual(Object.keys((await laptop('GET', '/api/progress')).body.done).sort(), ['gen-1', 'mx-1']);
});

test('deploy previews and branch deploys never touch the production store', () => {
  assert.equal(storeNameFor('production'), 'lemma');
  assert.equal(storeNameFor(undefined), 'lemma');
  assert.equal(storeNameFor('deploy-preview'), 'lemma-deploy-preview');
  assert.equal(storeNameFor('branch-deploy'), 'lemma-branch-deploy');
});
