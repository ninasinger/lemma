import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export const DEFAULT_DB_FILE = process.env.DATA_FILE ?? fileURLToPath(new URL('../data/db.json', import.meta.url));

const KINDS = ['users', 'sessions', 'progress', 'reviews', 'throttle'];

// A small JSON-file database for running on your own machine or server. Records live in one
// file grouped by kind ({ users: {…}, sessions: {…}, … }). Each write goes to a temp file that is
// then renamed over the real one, so a crash mid-write never leaves a half-written file behind.
// On Netlify, lib/blobstore.js provides the same get/put/del interface on Netlify Blobs.
export async function openStore(file) {
  let data;
  try {
    data = JSON.parse(await readFile(file, 'utf8'));
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
    data = {};
  }
  for (const kind of KINDS) data[kind] ??= {};

  const persist = async () => {
    await mkdir(dirname(file), { recursive: true });
    const tmp = `${file}.${process.pid}.tmp`;
    await writeFile(tmp, JSON.stringify(data, null, 2));
    await rename(tmp, file);
  };

  let queue = Promise.resolve();
  // Writes are queued so two requests can never interleave their renames.
  const save = () => {
    const run = queue.then(persist);
    queue = run.catch(() => {});
    return run;
  };

  return {
    data,
    async get(kind, key) {
      const value = data[kind][key];
      return value === undefined ? null : structuredClone(value);
    },
    async put(kind, key, value) {
      data[kind][key] = structuredClone(value);
      await save();
    },
    async del(kind, key) {
      if (!(key in data[kind])) return;
      delete data[kind][key];
      await save();
    },
    // Atomic here because one Node process owns the file: nothing runs between read and write.
    async update(kind, key, change) {
      const current = data[kind][key];
      const next = change(current === undefined ? null : structuredClone(current));
      data[kind][key] = structuredClone(next);
      await save();
      return structuredClone(next);
    },
    async keys(kind) {
      return Object.keys(data[kind]);
    },
  };
}
