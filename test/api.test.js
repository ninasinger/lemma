import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createApp } from '../server.js';
import { setPassword } from '../lib/auth.js';

const USER = 'tester';
const PASSWORD = 'correct horse battery';
let dir;
let dbFile;

before(async () => {
  dir = await mkdtemp(join(tmpdir(), 'lemma-test-'));
  dbFile = join(dir, 'db.json');
  const { store } = await createApp({ dbFile });
  await setPassword(store, USER, PASSWORD);
});

after(() => rm(dir, { recursive: true, force: true }));

async function start() {
  const app = await createApp({ dbFile });
  await new Promise((resolve) => app.server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${app.server.address().port}`;
  let cookie = '';
  const call = async (method, path, body, { contentType = 'application/json' } = {}) => {
    const headers = {};
    if (cookie) headers.Cookie = cookie;
    if (body !== undefined) headers['Content-Type'] = contentType;
    const res = await fetch(base + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
    const setCookie = res.headers.getSetCookie().find((c) => c.startsWith('sid='));
    if (setCookie) cookie = setCookie.split(';')[0] === 'sid=' ? '' : setCookie.split(';')[0];
    const text = await res.text();
    return { status: res.status, body: text ? JSON.parse(text) : null, raw: text, setCookie };
  };
  return { call, base, close: () => new Promise((resolve) => app.server.close(resolve)) };
}

test('sign in rejects a wrong password and accepts the right one', async () => {
  const app = await start();
  try {
    assert.equal((await app.call('POST', '/api/login', { username: USER, password: 'nope' })).status, 401);
    assert.equal((await app.call('POST', '/api/login', { username: 'nobody', password: PASSWORD })).status, 401);
    const ok = await app.call('POST', '/api/login', { username: ' Tester ', password: PASSWORD });
    assert.equal(ok.status, 200);
    assert.match(ok.setCookie, /HttpOnly/);
    assert.match(ok.setCookie, /SameSite=Lax/);
    assert.deepEqual((await app.call('GET', '/api/me')).body, { username: USER });
  } finally {
    await app.close();
  }
});

test('progress endpoints require a session', async () => {
  const app = await start();
  try {
    assert.equal((await app.call('GET', '/api/progress')).status, 401);
    assert.equal((await app.call('PUT', '/api/progress', { xp: 5 })).status, 401);
    assert.equal((await app.call('GET', '/api/me')).status, 401);
  } finally {
    await app.close();
  }
});

test('saved progress survives a server restart', async () => {
  const first = await start();
  await first.call('POST', '/api/login', { username: USER, password: PASSWORD });
  const progress = { done: { 'pca-1': true }, xp: 15, streak: 1, last: '2026-10-21', muted: false };
  const put = await first.call('PUT', '/api/progress', progress);
  assert.equal(put.status, 200);
  assert.deepEqual(put.body, progress);
  await first.close();

  const onDisk = JSON.parse(await readFile(dbFile, 'utf8'));
  assert.deepEqual(onDisk.progress[USER], progress);
  assert.ok(!JSON.stringify(onDisk).includes(PASSWORD), 'password is stored only as a hash');

  const second = await start();
  try {
    await second.call('POST', '/api/login', { username: USER, password: PASSWORD });
    assert.deepEqual((await second.call('GET', '/api/progress')).body, progress);
  } finally {
    await second.close();
  }
});

test('a stale browser tab cannot erase finished lessons', async () => {
  const app = await start();
  try {
    await app.call('POST', '/api/login', { username: USER, password: PASSWORD });
    await app.call('PUT', '/api/progress', { done: { 'pca-1': true, 'pca-2': true }, xp: 30, streak: 1, last: '2026-10-21' });
    const stale = await app.call('PUT', '/api/progress', { done: { 'svd-1': true }, xp: 10, streak: 1, last: '2026-10-21' });
    assert.deepEqual(Object.keys(stale.body.done).sort(), ['pca-1', 'pca-2', 'svd-1']);
    assert.equal(stale.body.xp, 30);
  } finally {
    await app.close();
  }
});

const cardState = (last, extra = {}) => ({ phase: 'review', step: 0, ivl: 3, ease: 2.5, reps: 2, lapses: 0, due: last + 3 * 86_400_000, last, intro: last - 60_000, ...extra });

test('review card states are saved and survive a restart', async () => {
  const first = await start();
  await first.call('POST', '/api/login', { username: USER, password: PASSWORD });
  const put = await first.call('PUT', '/api/reviews', { cards: { 'mx-dot-def': cardState(1_000_000) } });
  assert.deepEqual(put.body, { saved: 1 });
  await first.close();

  const second = await start();
  try {
    await second.call('POST', '/api/login', { username: USER, password: PASSWORD });
    const got = await second.call('GET', '/api/reviews');
    assert.deepEqual(got.body.cards['mx-dot-def'], cardState(1_000_000));
  } finally {
    await second.close();
  }
});

test('an older review never overwrites a newer one', async () => {
  const app = await start();
  try {
    await app.call('POST', '/api/login', { username: USER, password: PASSWORD });
    await app.call('PUT', '/api/reviews', { cards: { 'np-axis0': cardState(5_000_000, { ivl: 9 }) } });
    await app.call('PUT', '/api/reviews', { cards: { 'np-axis0': cardState(4_000_000, { ivl: 1 }), 'np-axis1': cardState(4_000_000) } });
    const { cards } = (await app.call('GET', '/api/reviews')).body;
    assert.equal(cards['np-axis0'].ivl, 9);
    assert.ok(cards['np-axis1']);
  } finally {
    await app.close();
  }
});

test('malformed review data is rejected as a whole', async () => {
  const app = await start();
  try {
    await app.call('POST', '/api/login', { username: USER, password: PASSWORD });
    const bad = [
      { cards: { 'Bad Id!': cardState(1) } },
      { cards: { 'ok-id': { ...cardState(1), phase: 'graduated' } } },
      { cards: { 'ok-id': { ...cardState(1), ease: 'high' } } },
      { cards: [] },
    ];
    for (const body of bad) assert.equal((await app.call('PUT', '/api/reviews', body)).status, 400, JSON.stringify(body));
    assert.equal((await app.call('GET', '/api/reviews')).body.cards['ok-id'], undefined);
  } finally {
    await app.close();
  }
});

test('reset clears lessons and reviews but keeps the sound setting', async () => {
  const app = await start();
  try {
    await app.call('POST', '/api/login', { username: USER, password: PASSWORD });
    await app.call('PUT', '/api/progress', { done: { 'dr-1': true }, xp: 99, muted: true });
    await app.call('PUT', '/api/reviews', { cards: { 'dr-def': cardState(1_000) } });
    const reset = await app.call('DELETE', '/api/progress');
    assert.deepEqual(reset.body, { done: {}, xp: 0, streak: 0, last: null, muted: true });
    assert.deepEqual((await app.call('GET', '/api/reviews')).body, { cards: {} });
  } finally {
    await app.close();
  }
});

test('writes must be JSON and well-formed', async () => {
  const app = await start();
  try {
    await app.call('POST', '/api/login', { username: USER, password: PASSWORD });
    assert.equal((await app.call('PUT', '/api/progress', { xp: 1 }, { contentType: 'text/plain' })).status, 415);
    assert.equal((await app.call('PUT', '/api/progress', { xp: 'lots' })).status, 400);
    assert.equal((await app.call('PUT', '/api/progress', { done: { ['x'.repeat(50)]: true } })).status, 400);
  } finally {
    await app.close();
  }
});

test('sign out ends the session', async () => {
  const app = await start();
  try {
    await app.call('POST', '/api/login', { username: USER, password: PASSWORD });
    assert.equal((await app.call('GET', '/api/me')).status, 200);
    const out = await app.call('POST', '/api/logout');
    assert.equal(out.status, 204);
    assert.match(out.setCookie, /Max-Age=0/);
    assert.equal((await app.call('GET', '/api/me')).status, 401);
  } finally {
    await app.close();
  }
});

test('repeated wrong passwords are locked out', async () => {
  const app = await start();
  try {
    const statuses = [];
    for (let i = 0; i < 11; i++) statuses.push((await app.call('POST', '/api/login', { username: USER, password: 'guess' + i })).status);
    assert.equal(statuses.at(-1), 429);
    assert.equal(statuses.filter((s) => s === 401).length, 10);
  } finally {
    await app.close();
  }
});

test('serves the app page but nothing outside the public folder', async () => {
  const app = await start();
  try {
    const page = await fetch(app.base + '/');
    assert.equal(page.status, 200);
    assert.match(page.headers.get('content-type'), /text\/html/);
    assert.match(await page.text(), /<title>Lemma<\/title>/);
    assert.match(page.headers.get('content-security-policy'), /script-src 'self'/);
    const manifest = await fetch(app.base + '/manifest.webmanifest');
    assert.match(manifest.headers.get('content-type'), /application\/manifest\+json/);
    assert.equal((await manifest.json()).display, 'standalone');
    for (const path of ['/sw.js', '/js/app.js', '/icons/icon-512.png', '/icons/maskable-512.png']) {
      assert.equal((await fetch(app.base + path)).status, 200, path);
    }
    assert.equal((await fetch(app.base + '/%2e%2e%2fserver.js')).status, 404);
    assert.equal((await fetch(app.base + '/..%2f.env')).status, 404);
  } finally {
    await app.close();
  }
});
