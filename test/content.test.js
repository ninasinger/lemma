import { test } from 'node:test';
import assert from 'node:assert/strict';
import { UNITS } from '../public/js/content/index.js';
import { W } from '../public/js/widgets.js';
import { parseShape, rich } from '../public/js/util.js';

const lessons = UNITS.flatMap((u) => u.lessons);
const cards = lessons.flatMap((l) => l.cards ?? []);
const STEP_TYPES = new Set(['learn', 'task', 'mc', 'tf', 'num', 'shape', 'match']);

test('unit, lesson and card ids are unique and permanent-safe', () => {
  const ids = [...UNITS.map((u) => u.id), ...lessons.map((l) => l.id)];
  assert.equal(new Set(ids).size, ids.length, 'duplicate unit or lesson id');
  assert.equal(new Set(cards.map((c) => c.id)).size, cards.length, 'duplicate card id');
  for (const id of [...lessons.map((l) => l.id), ...cards.map((c) => c.id)]) assert.match(id, /^[a-z0-9-]{1,40}$/);
});

test('every unit has a color token and a guidebook', () => {
  for (const u of UNITS) {
    assert.ok(u.c && u.title && u.sub && u.guide, u.id);
  }
});

test('every step is well formed', () => {
  for (const l of lessons) {
    for (const s of l.steps) {
      const where = `${l.id}: ${s.q ?? s.title}`;
      assert.ok(STEP_TYPES.has(s.t), `unknown type in ${where}`);
      if (s.w) assert.equal(typeof W[s.w], 'function', `unknown widget "${s.w}" in ${where}`);
      if (s.t === 'task') assert.ok(s.w && s.hint && s.ex, where);
      if (s.t === 'mc') assert.ok(s.o.length >= 2 && s.ex, where);
      if (s.t === 'tf') assert.equal(typeof s.a, 'boolean', where);
      if (s.t === 'num') assert.ok(Number.isFinite(s.a), where);
      if (s.t === 'shape') {
        assert.ok(parseShape(s.a).length || s.a === '()', where);
        if (s.a === 'error') assert.ok(s.err, `${where}: an "error" answer needs err: true so the hint mentions it`);
      }
      if (s.t === 'match') assert.ok(s.p.length >= 2 && s.p.every((p) => p.length === 2), where);
    }
  }
});

test('every lesson unlocks at least three review cards', () => {
  for (const l of lessons) assert.ok((l.cards ?? []).length >= 3, l.id);
  for (const c of cards) assert.ok(c.front && c.back, c.id);
});

test('shape answers parse the way people type them', () => {
  assert.deepEqual(parseShape('(500, 3)'), ['500', '3']);
  assert.deepEqual(parseShape('500,3'), ['500', '3']);
  assert.deepEqual(parseShape('[N, D]'), ['n', 'd']);
  assert.deepEqual(parseShape('(3,)'), ['3']);
  assert.deepEqual(parseShape('()'), []);
  assert.deepEqual(parseShape('Error'), ['error']);
});

test('rich text turns {{…}} into a matrix', () => {
  const html = rich('A = {{1 2; 3 4}}');
  assert.match(html, /class="mtx" style="--cols:2"/);
  assert.equal((html.match(/<span>/g) || []).length, 4);
});
