// The JSON API, written against the web-standard Request and Response so the same code runs in
// the Node server (server.js) and in a Netlify Function (netlify/functions/api.mjs).
//
// `store` is any object with async get(kind, key), put(kind, key, value), del(kind, key) and
// update(kind, key, change), where update applies change(current) as one atomic read-modify-write.
// An optional keys(kind) lets expired sessions be pruned. See lib/store.js and lib/blobstore.js.

import { hashPassword, hashToken, isValidUsername, newSessionToken, normalizeUsername, setPassword, verifyPassword } from './auth.js';
import { emptyProgress, mergeProgress, sanitizeProgress } from './progress.js';
import { mergeReviews, sanitizeReviewBatch } from './reviews.js';

const SESSION_DAYS = 30;
const MAX_BODY_BYTES = 16 * 1024;
export const MAX_REVIEW_BODY_BYTES = 512 * 1024;
const MAX_LOGIN_FAILURES = 10;
const LOCKOUT_MS = 15 * 60 * 1000;

export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const json = (status, body, headers = {}) =>
  new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', ...headers },
  });

function parseCookies(header) {
  const cookies = {};
  for (const part of (header ?? '').split(';')) {
    const eq = part.indexOf('=');
    if (eq > 0) cookies[part.slice(0, eq).trim()] = part.slice(eq + 1).trim();
  }
  return cookies;
}

async function readJson(request, maxBytes = MAX_BODY_BYTES) {
  // Requiring a JSON content type also stops other sites from posting plain HTML forms here.
  if (!/^application\/json\b/i.test(request.headers.get('content-type') ?? '')) {
    throw new HttpError(415, 'Send JSON with Content-Type: application/json.');
  }
  const text = await request.text();
  if (Buffer.byteLength(text) > maxBytes) throw new HttpError(413, 'Request body is too large.');
  try {
    return JSON.parse(text || '{}');
  } catch {
    throw new HttpError(400, 'Request body is not valid JSON.');
  }
}

// secureCookies: true or false to force it; leave undefined to follow the request's protocol.
// seedUser: { username, password } creates that account on first use if it doesn't exist yet.
export function createApi({ store, secureCookies, seedUser } = {}) {
  // Checked against when the username doesn't exist, so a wrong username takes as long as a wrong password.
  const decoyHash = hashPassword('decoy');

  let seeding = null;
  function ensureSeedUser() {
    if (!seedUser?.username || !seedUser?.password) return null;
    seeding ??= (async () => {
      if (!(await store.get('users', normalizeUsername(seedUser.username)))) {
        await setPassword(store, seedUser.username, seedUser.password);
      }
    })().catch((err) => {
      seeding = null;
      throw err;
    });
    return seeding;
  }

  const sessionCookie = (request, value, maxAgeSeconds) => {
    const secure = secureCookies ?? new URL(request.url).protocol === 'https:';
    return `sid=${value}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${maxAgeSeconds}${secure ? '; Secure' : ''}`;
  };

  async function currentUser(request) {
    const token = parseCookies(request.headers.get('cookie')).sid;
    if (!token) return null;
    const session = await store.get('sessions', hashToken(token));
    if (!session || session.expires < Date.now()) return null;
    return (await store.get('users', session.username)) ? session.username : null;
  }

  async function requireUser(request) {
    const username = await currentUser(request);
    if (!username) throw new HttpError(401, 'Sign in to continue.');
    return username;
  }

  async function pruneExpiredSessions() {
    if (!store.keys) return;
    const now = Date.now();
    for (const key of await store.keys('sessions')) {
      const session = await store.get('sessions', key);
      if (session && session.expires < now) await store.del('sessions', key);
    }
  }

  const routes = {
    async 'POST /api/login'(request, { clientIp }) {
      const throttleKey = hashToken(`login:${clientIp || 'unknown'}`);
      const record = await store.get('throttle', throttleKey);
      const locked = record && record.until > Date.now();
      if (locked && record.count >= MAX_LOGIN_FAILURES) {
        throw new HttpError(429, 'Too many sign-in attempts. Wait 15 minutes and try again.');
      }

      const body = await readJson(request);
      const username = normalizeUsername(body.username);
      const user = isValidUsername(username) ? await store.get('users', username) : null;
      const ok = await verifyPassword(String(body.password ?? ''), user ? user.password : await decoyHash);
      if (!user || !ok) {
        await store.update('throttle', throttleKey, (cur) => {
          const now = Date.now();
          return { count: (cur && cur.until > now ? cur.count : 0) + 1, until: now + LOCKOUT_MS };
        });
        throw new HttpError(401, 'That username and password don’t match.');
      }
      if (record) await store.del('throttle', throttleKey);

      await pruneExpiredSessions();
      const token = newSessionToken();
      await store.put('sessions', hashToken(token), { username, expires: Date.now() + SESSION_DAYS * 86_400_000 });
      return json(200, { username }, { 'Set-Cookie': sessionCookie(request, token, SESSION_DAYS * 86_400) });
    },

    async 'POST /api/logout'(request) {
      const token = parseCookies(request.headers.get('cookie')).sid;
      if (token) await store.del('sessions', hashToken(token));
      return json(204, undefined, { 'Set-Cookie': sessionCookie(request, '', 0) });
    },

    async 'GET /api/me'(request) {
      return json(200, { username: await requireUser(request) });
    },

    async 'GET /api/progress'(request) {
      const username = await requireUser(request);
      return json(200, (await store.get('progress', username)) ?? emptyProgress());
    },

    async 'PUT /api/progress'(request) {
      const username = await requireUser(request);
      const incoming = sanitizeProgress(await readJson(request));
      if (!incoming) throw new HttpError(400, 'Progress data is malformed.');
      const merged = await store.update('progress', username, (cur) => mergeProgress(cur ?? emptyProgress(), incoming));
      return json(200, merged);
    },

    // Resets lessons and review history together; keeps the sound setting.
    async 'DELETE /api/progress'(request) {
      const username = await requireUser(request);
      const muted = (await store.get('progress', username))?.muted ?? false;
      const fresh = { ...emptyProgress(), muted };
      await store.put('progress', username, fresh);
      await store.del('reviews', username);
      return json(200, fresh);
    },

    async 'GET /api/reviews'(request) {
      const username = await requireUser(request);
      return json(200, { cards: (await store.get('reviews', username)) ?? {} });
    },

    async 'PUT /api/reviews'(request) {
      const username = await requireUser(request);
      const incoming = sanitizeReviewBatch(await readJson(request, MAX_REVIEW_BODY_BYTES));
      if (!incoming) throw new HttpError(400, 'Review data is malformed.');
      await store.update('reviews', username, (cur) => mergeReviews(cur ?? {}, incoming));
      return json(200, { saved: Object.keys(incoming).length });
    },
  };

  return {
    async handle(request, context = {}) {
      try {
        const { pathname } = new URL(request.url);
        const route = routes[`${request.method} ${pathname}`];
        if (!route) throw new HttpError(404, 'Not found.');
        await ensureSeedUser();
        return await route(request, context);
      } catch (err) {
        const status = err instanceof HttpError ? err.status : 500;
        if (status === 500) console.error(err);
        return json(status, { error: status === 500 ? 'Something went wrong on the server.' : err.message });
      }
    },
  };
}
