import { createHash, randomBytes, scrypt as scryptCallback, timingSafeEqual } from 'node:crypto';
import { promisify } from 'node:util';

const scrypt = promisify(scryptCallback);
const KEY_LENGTH = 64;

export async function hashPassword(password) {
  const salt = randomBytes(16);
  const key = await scrypt(password, salt, KEY_LENGTH);
  return `scrypt$${salt.toString('base64')}$${key.toString('base64')}`;
}

export async function verifyPassword(password, stored) {
  const [scheme, salt, key] = String(stored).split('$');
  if (scheme !== 'scrypt' || !salt || !key) return false;
  const expected = Buffer.from(key, 'base64');
  const actual = await scrypt(password, Buffer.from(salt, 'base64'), expected.length);
  return timingSafeEqual(expected, actual);
}

export const newSessionToken = () => randomBytes(32).toString('base64url');

// Only a hash of each session token is stored, so a copied data file can't be used to sign in.
export const hashToken = (token) => createHash('sha256').update(token).digest('hex');

export const normalizeUsername = (name) => String(name ?? '').trim().toLowerCase();

export const isValidUsername = (name) => /^[a-z0-9_.-]{2,32}$/.test(name);

// Creates the user, or replaces the password of an existing one. Returns true if the user is new.
export async function setPassword(store, rawUsername, password) {
  const username = normalizeUsername(rawUsername);
  if (!isValidUsername(username)) {
    throw new Error('Usernames are 2–32 characters: lowercase letters, digits, dots, dashes or underscores.');
  }
  if (typeof password !== 'string' || password.length === 0) {
    throw new Error('Password must not be empty.');
  }
  const existing = store.data.users[username];
  store.data.users[username] = {
    username,
    password: await hashPassword(password),
    createdAt: existing?.createdAt ?? new Date().toISOString(),
  };
  await store.save();
  return !existing;
}
