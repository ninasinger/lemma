import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export const DEFAULT_DB_FILE = process.env.DATA_FILE ?? fileURLToPath(new URL('../data/db.json', import.meta.url));

const empty = () => ({ users: {}, sessions: {}, progress: {}, reviews: {} });

// A small JSON-file database. Each write goes to a temp file that is then renamed
// over the real one, so a crash mid-write never leaves a half-written file behind.
export async function openStore(file) {
  let data;
  try {
    data = { ...empty(), ...JSON.parse(await readFile(file, 'utf8')) };
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
    data = empty();
  }

  const persist = async () => {
    await mkdir(dirname(file), { recursive: true });
    const tmp = `${file}.${process.pid}.tmp`;
    await writeFile(tmp, JSON.stringify(data, null, 2));
    await rename(tmp, file);
  };

  let queue = Promise.resolve();
  return {
    data,
    // Writes are queued so two requests can never interleave their renames.
    save() {
      const run = queue.then(persist);
      queue = run.catch(() => {});
      return run;
    },
  };
}
