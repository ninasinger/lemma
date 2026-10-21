// Builds the Netlify Function that serves /api/*. Kept separate from netlify/functions/api.mjs so
// tests can pass in a fake Blobs store instead of the real one.
import { createApi } from './api.js';
import { blobStore } from './blobstore.js';

// Production uses the store "lemma". Deploy previews and branch deploys get their own store, so a
// preview can never read or overwrite your real progress.
export const storeNameFor = (deployContext) =>
  !deployContext || deployContext === 'production' ? 'lemma' : `lemma-${deployContext}`;

// openBlobs(name) opens the named Blobs store with the current request's credentials. It runs for
// every store operation, because those credentials expire while the function instance stays warm.
// readEnv(key) reads a setting: SEED_USERNAME and SEED_PASSWORD (set in the Netlify UI) create
// that account if it doesn't exist yet.
export function netlifyHandler(openBlobs, readEnv = (key) => process.env[key]) {
  let api;
  return async (request, context) => {
    const name = storeNameFor(context?.deploy?.context);
    api ??= createApi({
      store: blobStore(() => openBlobs(name)),
      seedUser: { username: readEnv('SEED_USERNAME'), password: readEnv('SEED_PASSWORD') },
    });
    return api.handle(request, { clientIp: context?.ip });
  };
}
