import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const ORIGIN = 'https://lemma.example';

// Runs public/sw.js against an in-memory Cache Storage and a fake network.
async function loadWorker({ online = true } = {}) {
  const stores = new Map();
  const open = (name) => {
    if (!stores.has(name)) stores.set(name, new Map());
    const m = stores.get(name);
    const key = (r) => new URL(typeof r === 'string' ? r : r.url, ORIGIN).pathname;
    return {
      addAll: async (urls) => urls.forEach((u) => m.set(key(u), new Response(`cached ${u}`))),
      put: async (r, res) => m.set(key(r), res),
      match: async (r) => m.get(key(r))?.clone(),
    };
  };
  const listeners = {};
  const net = { online, calls: [] };
  const ctx = {
    self: {
      location: { origin: ORIGIN },
      addEventListener: (type, fn) => (listeners[type] = fn),
      skipWaiting: async () => {},
      clients: { claim: async () => {} },
    },
    caches: { open: async (n) => open(n), keys: async () => [...stores.keys()], delete: async (n) => stores.delete(n) },
    fetch: async (req) => {
      net.calls.push(new URL(req.url ?? req, ORIGIN).pathname);
      if (!net.online) throw new TypeError('offline');
      return new Response(`fresh ${new URL(req.url ?? req, ORIGIN).pathname}`);
    },
    Response,
    URL,
  };
  ctx.self.caches = ctx.caches;
  vm.runInNewContext(await readFile(new URL('../public/sw.js', import.meta.url), 'utf8'), ctx);

  const fire = async (type, init = {}) => {
    let responded;
    let waited = Promise.resolve();
    listeners[type]({ ...init, waitUntil: (p) => (waited = p), respondWith: (p) => (responded = p) });
    await waited;
    return responded;
  };
  const get = (path, mode = 'cors') => fire('fetch', { request: { method: 'GET', url: ORIGIN + path, mode } });
  return { fire, get, net, stores };
}

test('install precaches the app shell', async () => {
  const sw = await loadWorker();
  await sw.fire('install');
  const cache = [...sw.stores.values()][0];
  for (const path of ['/', '/css/app.css', '/js/app.js', '/manifest.webmanifest']) assert.ok(cache.has(path), path);
});

test('API calls are never intercepted', async () => {
  const sw = await loadWorker();
  assert.equal(await sw.get('/api/progress'), undefined);
});

test('app files come from the network first and are cached as they load', async () => {
  const sw = await loadWorker();
  const res = await sw.get('/js/content/pca.js');
  assert.equal(await res.text(), 'fresh /js/content/pca.js');
  sw.net.online = false;
  assert.equal(await (await sw.get('/js/content/pca.js')).text(), 'fresh /js/content/pca.js');
});

test('offline navigation falls back to the cached app shell', async () => {
  const sw = await loadWorker();
  await sw.fire('install');
  sw.net.online = false;
  const res = await sw.get('/some/deep/link', 'navigate');
  assert.equal(await res.text(), 'cached /');
});

test('old caches are deleted on activate', async () => {
  const sw = await loadWorker();
  await sw.fire('install');
  sw.stores.set('lemma-v0', new Map());
  await sw.fire('activate');
  assert.deepEqual([...sw.stores.keys()], ['lemma-v1']);
});
