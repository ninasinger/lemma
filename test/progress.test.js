import { test } from 'node:test';
import assert from 'node:assert/strict';
import { emptyProgress, mergeProgress, sanitizeProgress } from '../lib/progress.js';

test('sanitizeProgress keeps a well-formed object and drops unfinished lessons', () => {
  const clean = sanitizeProgress({ done: { 'pca-1': true, 'svd-1': false }, xp: 25, streak: 2, last: '2026-10-21', muted: true });
  assert.deepEqual(clean, { done: { 'pca-1': true }, xp: 25, streak: 2, last: '2026-10-21', muted: true });
});

test('sanitizeProgress rejects malformed input', () => {
  for (const bad of [
    null,
    [],
    'progress',
    { done: [] },
    { done: { '../etc': true } },
    { xp: -1 },
    { xp: 1.5 },
    { streak: '3' },
    { last: 'yesterday' },
    { muted: 'yes' },
  ]) {
    assert.equal(sanitizeProgress(bad), null, JSON.stringify(bad));
  }
});

test('sanitizeProgress fills in defaults for missing fields', () => {
  assert.deepEqual(sanitizeProgress({}), emptyProgress());
});

test('mergeProgress unions lessons and keeps the higher XP', () => {
  const saved = { done: { 'pca-1': true, 'pca-2': true }, xp: 40, streak: 1, last: '2026-10-20', muted: false };
  const incoming = { done: { 'svd-1': true }, xp: 10, streak: 1, last: '2026-10-20', muted: true };
  const merged = mergeProgress(saved, incoming);
  assert.deepEqual(merged.done, { 'pca-1': true, 'pca-2': true, 'svd-1': true });
  assert.equal(merged.xp, 40);
  assert.equal(merged.muted, true);
});

test('mergeProgress takes the streak from whichever side played most recently', () => {
  const older = { ...emptyProgress(), streak: 5, last: '2026-10-19' };
  const newer = { ...emptyProgress(), streak: 1, last: '2026-10-21' };
  assert.deepEqual([mergeProgress(older, newer).streak, mergeProgress(older, newer).last], [1, '2026-10-21']);
  assert.deepEqual([mergeProgress(newer, older).streak, mergeProgress(newer, older).last], [1, '2026-10-21']);
});
