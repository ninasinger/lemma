// Node server for running Lemma on your own machine or a server with a disk.
// It serves public/ and passes /api/* to the shared handler in lib/api.js.
// On Netlify, the CDN serves public/ and netlify/functions/api.mjs runs the same handler.

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_DB_FILE, openStore } from './lib/store.js';
import { createApi, HttpError, MAX_REVIEW_BODY_BYTES } from './lib/api.js';

const ROOT = fileURLToPath(new URL('.', import.meta.url));

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
// netlify.toml sends the same policy; test/netlify.test.js keeps them in sync.
export const CONTENT_SECURITY_POLICY = [
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

// Request headers that describe this one connection rather than the request itself.
const HOP_BY_HOP = new Set(['host', 'connection', 'keep-alive', 'content-length', 'transfer-encoding', 'upgrade']);

export async function createApp({ dbFile = DEFAULT_DB_FILE, publicDir = join(ROOT, 'public'), secureCookies = false, trustProxy = false, seedUser } = {}) {
  const store = await openStore(dbFile);
  const api = createApi({ store, secureCookies, seedUser });
  // Behind a reverse proxy every request arrives from the proxy, so use the address it forwards.
  const clientIp = (req) =>
    (trustProxy && String(req.headers['x-forwarded-for'] ?? '').split(',')[0].trim()) || req.socket.remoteAddress || '';

  const server = createServer(async (req, res) => {
    try {
      const { pathname } = new URL(req.url, 'http://localhost');
      if (pathname.startsWith('/api/')) {
        const response = await api.handle(await toWebRequest(req), { clientIp: clientIp(req) });
        res.writeHead(response.status, Object.fromEntries(response.headers));
        res.end(Buffer.from(await response.arrayBuffer()));
        return;
      }
      if (req.method !== 'GET' && req.method !== 'HEAD') throw new HttpError(405, 'Method not allowed.');
      await serveStatic(publicDir, pathname, req, res);
    } catch (err) {
      const status = err instanceof HttpError ? err.status : 500;
      if (status === 500) console.error(err);
      if (res.headersSent) return res.end();
      res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify({ error: status === 500 ? 'Something went wrong on the server.' : err.message }));
    }
  });

  return { server, store };
}

async function toWebRequest(req) {
  const headers = new Headers();
  for (const [name, value] of Object.entries(req.headers)) {
    if (value !== undefined && !HOP_BY_HOP.has(name)) headers.set(name, Array.isArray(value) ? value.join(', ') : value);
  }
  let body;
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
      size += chunk.length;
      if (size > MAX_REVIEW_BODY_BYTES) throw new HttpError(413, 'Request body is too large.');
      chunks.push(chunk);
    }
    body = Buffer.concat(chunks);
  }
  return new Request(new URL(req.url, 'http://localhost'), { method: req.method, headers, body });
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
  const { SEED_USERNAME: username, SEED_PASSWORD: password } = process.env;
  const { server, store } = await createApp({
    secureCookies: process.env.NODE_ENV === 'production',
    trustProxy: process.env.TRUST_PROXY === '1',
    seedUser: username && password ? { username, password } : undefined,
  });
  if (Object.keys(store.data.users).length === 0 && !(username && password)) {
    console.warn('No users yet. Copy .env.example to .env, set a username and password, then run `npm run seed`.');
  }
  server.listen(port, host, () => console.log(`Lemma is running at http://${host === '0.0.0.0' ? 'localhost' : host}:${port}`));
}
