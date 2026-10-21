// Storage on Netlify Blobs, with the same interface as the file store in lib/store.js.
// Each record is its own JSON blob, keyed "<kind>/<key>", e.g. "progress/nina".
// openBlobs() returns the Blobs store, e.g. getStore({ name, consistency: 'strong' }) so a write
// is visible to the very next request, even when it lands on a different function instance.
// It's called for every operation, not once: a store keeps the short-lived token of the request
// it was opened in, and a warm function instance outlives that token.
const MAX_ATTEMPTS = 8;
const BACKOFF_BASE_MS = 20;
const BACKOFF_MAX_MS = 1000;

// After a conflict, wait a random time that grows with each attempt, so writers that keep colliding
// spread out instead of retrying in lockstep until one runs out of attempts.
const backoff = (attempt) =>
  new Promise((resolve) => setTimeout(resolve, Math.random() * Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** attempt)));

export function blobStore(openBlobs) {
  const id = (kind, key) => `${kind}/${key}`;
  return {
    async get(kind, key) {
      return (await openBlobs().get(id(kind, key), { type: 'json' })) ?? null;
    },
    async put(kind, key, value) {
      await openBlobs().setJSON(id(kind, key), value);
    },
    async del(kind, key) {
      await openBlobs().delete(id(kind, key));
    },
    // Read-modify-write without lost updates: write only if the blob is unchanged since we read
    // it (its ETag still matches, or it still doesn't exist); otherwise re-read and try again.
    async update(kind, key, change) {
      const name = id(kind, key);
      const blobs = openBlobs();
      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        const current = await blobs.getWithMetadata(name, { type: 'json' });
        const next = change(current ? current.data : null);
        const { modified } = current
          ? await blobs.setJSON(name, next, { onlyIfMatch: current.etag })
          : await blobs.setJSON(name, next, { onlyIfNew: true });
        if (modified) return next;
        if (attempt < MAX_ATTEMPTS - 1) await backoff(attempt);
      }
      throw new Error(`Gave up updating ${name} after ${MAX_ATTEMPTS} conflicting writes.`);
    },
  };
}
