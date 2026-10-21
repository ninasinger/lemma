import { test } from 'node:test';
import assert from 'node:assert/strict';
import { hashPassword, normalizeUsername, verifyPassword } from '../lib/auth.js';

test('a hashed password verifies only with the same password', async () => {
  const stored = await hashPassword('correct horse');
  assert.match(stored, /^scrypt\$/);
  assert.ok(!stored.includes('correct horse'));
  assert.equal(await verifyPassword('correct horse', stored), true);
  assert.equal(await verifyPassword('wrong horse', stored), false);
});

test('two hashes of the same password use different salts', async () => {
  assert.notEqual(await hashPassword('same'), await hashPassword('same'));
});

test('verifyPassword rejects stored values in an unknown format', async () => {
  assert.equal(await verifyPassword('x', 'plaintext'), false);
  assert.equal(await verifyPassword('x', 'md5$abc$def'), false);
});

test('usernames are trimmed and lowercased', () => {
  assert.equal(normalizeUsername('  Nina '), 'nina');
});
