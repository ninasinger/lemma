// Creates (or resets the password of) the user named in SEED_USERNAME / SEED_PASSWORD.
// Run with `npm run seed`, which loads those values from .env.
import { DEFAULT_DB_FILE, openStore } from '../lib/store.js';
import { normalizeUsername, setPassword } from '../lib/auth.js';

const { SEED_USERNAME: username, SEED_PASSWORD: password } = process.env;
if (!username || !password) {
  console.error('Set SEED_USERNAME and SEED_PASSWORD (copy .env.example to .env), then run `npm run seed` again.');
  process.exit(1);
}

const store = await openStore(DEFAULT_DB_FILE);
const created = await setPassword(store, username, password);
console.log(`${created ? 'Created user' : 'Updated the password for'} "${normalizeUsername(username)}" in ${DEFAULT_DB_FILE}`);
