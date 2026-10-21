// Storage on Netlify Blobs, with the same interface as the file store in lib/store.js.
// Each record is its own JSON blob, keyed "<kind>/<key>", e.g. "progress/nina".
// Pass a store from getStore({ name, consistency: 'strong' }) so a write is visible to the
// very next request, even when it lands on a different function instance.
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
  };
}
