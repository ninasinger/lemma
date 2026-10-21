import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_DB_FILE, openStore } from './lib/store.js';
import { hashPassword, hashToken, newSessionToken, normalizeUsername, verifyPassword } from './lib/auth.js';
import { emptyProgress, mergeProgress, sanitizeProgress } from './lib/progress.js';
import { mergeReviews, sanitizeReviewBatch } from './lib/reviews.js';

const ROOT = fileURLToPath(new URL('.', import.meta.url));

const SESSION_DAYS = 30;
const MAX_BODY_BYTES = 16 * 1024;
const MAX_REVIEW_BODY_BYTES = 512 * 1024;
const MAX_LOGIN_FAILURES = 10;
const LOCKOUT_MS = 15 * 60 * 1000;
const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};
// Scripts may only come from this server; fonts come from Google Fonts.
const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com",
  "img-src 'self' data:",
  "connect-src 'self'",
  "manifest-src 'self'",
  "worker-src 'self'",
  "base-uri 'none'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join('; ');

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export async function createApp({ dbFile = DEFAULT_DB_FILE, publicDir = join(ROOT, 'public'), secureCookies = false, trustProxy = false } = {}) {
  const store = await openStore(dbFile);
  store.data.reviews ??= {};
  // Behind a reverse proxy every request arrives from the proxy, so use the address it forwards.
  const clientIp = (req) =>
    (trustProxy && String(req.headers['x-forwarded-for'] ?? '').split(',')[0].trim()) || req.socket.remoteAddress || '';
  const loginFailures = new Map();
  // Checked against when the username doesn't exist, so a wrong username takes as long as a wrong password.
  const decoyHash = await hashPassword('decoy');

  const sessionCookie = (value, maxAgeSeconds) =>
    `sid=${value}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${maxAgeSeconds}${secureCookies ? '; Secure' : ''}`;

  function currentUser(req) {
    const token = parseCookies(req.headers.cookie).sid;
    if (!token) return null;
    const session = store.data.sessions[hashToken(token)];
    if (!session || session.expires < Date.now()) return null;
    return store.data.users[session.username] ? session.username : null;
  }

  function requireUser(req) {
    const username = currentUser(req);
    if (!username) throw new HttpError(401, 'Sign in to continue.');
    return username;
  }

  function pruneExpiredSessions() {
    const now = Date.now();
    for (const [key, session] of Object.entries(store.data.sessions)) {
      if (session.expires < now) delete store.data.sessions[key];
    }
  }

  const routes = {
    async 'POST /api/login'(req, res) {
      const ip = clientIp(req);
      const record = loginFailures.get(ip);
      const locked = record && record.until > Date.now();
      if (locked && record.count >= MAX_LOGIN_FAILURES) {
        throw new HttpError(429, 'Too many sign-in attempts. Wait 15 minutes and try again.');
      }

      const body = await readJson(req);
      const username = normalizeUsername(body.username);
      const user = store.data.users[username];
      const password = String(body.password ?? '');
      const ok = await verifyPassword(password, user ? user.password : decoyHash);
      if (!user || !ok) {
        loginFailures.set(ip, { count: (locked ? record.count : 0) + 1, until: Date.now() + LOCKOUT_MS });
        throw new HttpError(401, 'That username and password don’t match.');
      }
      loginFailures.delete(ip);

      pruneExpiredSessions();
      const token = newSessionToken();
      store.data.sessions[hashToken(token)] = { username, expires: Date.now() + SESSION_DAYS * 86_400_000 };
      await store.save();
      res.setHeader('Set-Cookie', sessionCookie(token, SESSION_DAYS * 86_400));
      sendJson(res, 200, { username });
    },

    async 'POST /api/logout'(req, res) {
      const token = parseCookies(req.headers.cookie).sid;
      if (token && store.data.sessions[hashToken(token)]) {
        delete store.data.sessions[hashToken(token)];
        await store.save();
      }
      res.setHeader('Set-Cookie', sessionCookie('', 0));
      sendJson(res, 204);
    },

    async 'GET /api/me'(req, res) {
      sendJson(res, 200, { username: requireUser(req) });
    },

    async 'GET /api/progress'(req, res) {
      const username = requireUser(req);
      sendJson(res, 200, store.data.progress[username] ?? emptyProgress());
    },

    async 'PUT /api/progress'(req, res) {
      const username = requireUser(req);
      const incoming = sanitizeProgress(await readJson(req));
      if (!incoming) throw new HttpError(400, 'Progress data is malformed.');
      const merged = mergeProgress(store.data.progress[username] ?? emptyProgress(), incoming);
      store.data.progress[username] = merged;
      await store.save();
      sendJson(res, 200, merged);
    },

    // Resets lessons and review history together; keeps the sound setting.
    async 'DELETE /api/progress'(req, res) {
      const username = requireUser(req);
      const muted = store.data.progress[username]?.muted ?? false;
      store.data.progress[username] = { ...emptyProgress(), muted };
      delete store.data.reviews[username];
      await store.save();
      sendJson(res, 200, store.data.progress[username]);
    },

    async 'GET /api/reviews'(req, res) {
      const username = requireUser(req);
      sendJson(res, 200, { cards: store.data.reviews[username] ?? {} });
    },

    async 'PUT /api/reviews'(req, res) {
      const username = requireUser(req);
      const incoming = sanitizeReviewBatch(await readJson(req, MAX_REVIEW_BODY_BYTES));
      if (!incoming) throw new HttpError(400, 'Review data is malformed.');
      store.data.reviews[username] = mergeReviews(store.data.reviews[username] ?? {}, incoming);
      await store.save();
      sendJson(res, 200, { saved: Object.keys(incoming).length });
    },
  };

  const server = createServer(async (req, res) => {
    try {
      const { pathname } = new URL(req.url, 'http://localhost');
      if (pathname.startsWith('/api/')) {
        const route = routes[`${req.method} ${pathname}`];
        if (!route) throw new HttpError(404, 'Not found.');
        await route(req, res);
        return;
      }
      if (req.method !== 'GET' && req.method !== 'HEAD') throw new HttpError(405, 'Method not allowed.');
      await serveStatic(publicDir, pathname, req, res);
    } catch (err) {
      const status = err instanceof HttpError ? err.status : 500;
      if (status === 500) console.error(err);
      if (res.headersSent) return res.end();
      sendJson(res, status, { error: status === 500 ? 'Something went wrong on the server.' : err.message });
    }
  });

  return { server, store };
}

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(body === undefined ? undefined : JSON.stringify(body));
}

async function readJson(req, maxBytes = MAX_BODY_BYTES) {
  // Requiring a JSON content type also stops other sites from posting plain HTML forms here.
  if (!/^application\/json\b/i.test(req.headers['content-type'] ?? '')) {
    throw new HttpError(415, 'Send JSON with Content-Type: application/json.');
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) throw new HttpError(413, 'Request body is too large.');
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
  } catch {
    throw new HttpError(400, 'Request body is not valid JSON.');
  }
}

function parseCookies(header = '') {
  const cookies = {};
  for (const part of header.split(';')) {
    const eq = part.indexOf('=');
    if (eq > 0) cookies[part.slice(0, eq).trim()] = part.slice(eq + 1).trim();
  }
  return cookies;
}

async function serveStatic(publicDir, pathname, req, res) {
  let relative;
  try {
    relative = pathname === '/' ? 'index.html' : decodeURIComponent(pathname).replace(/^\/+/, '');
  } catch {
    throw new HttpError(400, 'Malformed URL.');
  }
  const root = resolve(publicDir);
  const file = resolve(root, relative);
  if (!file.startsWith(root + sep)) throw new HttpError(404, 'Not found.');

  let body;
  try {
    body = await readFile(file);
  } catch (err) {
    if (err.code === 'ENOENT' || err.code === 'EISDIR') throw new HttpError(404, 'Not found.');
    throw err;
  }
  const type = CONTENT_TYPES[extname(file)] ?? 'application/octet-stream';
  const headers = {
    'Content-Type': type,
    'Cache-Control': 'no-cache',
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'same-origin',
  };
  if (type.startsWith('text/html')) {
    headers['Content-Security-Policy'] = CONTENT_SECURITY_POLICY;
    headers['X-Frame-Options'] = 'DENY';
  }
  res.writeHead(200, headers);
  res.end(req.method === 'HEAD' ? undefined : body);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT) || 3000;
  const host = process.env.HOST ?? '127.0.0.1';
  const { server, store } = await createApp({
    secureCookies: process.env.NODE_ENV === 'production',
    trustProxy: process.env.TRUST_PROXY === '1',
  });
  if (Object.keys(store.data.users).length === 0) {
    console.warn('No users yet. Copy .env.example to .env, set a username and password, then run `npm run seed`.');
  }
  server.listen(port, host, () => console.log(`Lemma is running at http://${host === '0.0.0.0' ? 'localhost' : host}:${port}`));
}
