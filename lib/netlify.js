// Builds the Netlify Function that serves /api/*. Kept separate from netlify/functions/api.mjs so
// tests can pass in a fake Blobs store instead of the real one.
import { createApi } from './api.js';
import { blobStore } from './blobstore.js';

// openBlobs is called once, on the first request, when Netlify has provided the Blobs credentials.
// SEED_USERNAME and SEED_PASSWORD (set in the Netlify UI) create that account if it doesn't exist.
export function netlifyHandler(openBlobs, env = process.env) {
  let api;
  return async (request, context) => {
    api ??= createApi({
      store: blobStore(openBlobs()),
      seedUser: { username: env.SEED_USERNAME, password: env.SEED_PASSWORD },
    });
    return api.handle(request, { clientIp: context?.ip });
  };
}
