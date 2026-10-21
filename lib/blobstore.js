// Storage on Netlify Blobs, with the same interface as the file store in lib/store.js.
// Each record is its own JSON blob, keyed "<kind>/<key>", e.g. "progress/nina".
// Pass a store from getStore({ name, consistency: 'strong' }) so a write is visible to the
// very next request, even when it lands on a different function instance.
const MAX_ATTEMPTS = 8;

export function blobStore(blobs) {
  const id = (kind, key) => `${kind}/${key}`;
  return {
    async get(kind, key) {
      return (await blobs.get(id(kind, key), { type: 'json' })) ?? null;
    },
    async put(kind, key, value) {
      await blobs.setJSON(id(kind, key), value);
    },
    async del(kind, key) {
      await blobs.delete(id(kind, key));
    },
    // Read-modify-write without lost updates: write only if the blob is unchanged since we read
    // it (its ETag still matches, or it still doesn't exist); otherwise re-read and try again.
    async update(kind, key, change) {
      const name = id(kind, key);
      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        const current = await blobs.getWithMetadata(name, { type: 'json' });
        const next = change(current ? current.data : null);
        const { modified } = current
          ? await blobs.setJSON(name, next, { onlyIfMatch: current.etag })
          : await blobs.setJSON(name, next, { onlyIfNew: true });
        if (modified) return next;
      }
      throw new Error(`Gave up updating ${name} after ${MAX_ATTEMPTS} conflicting writes.`);
    },
  };
}
