import { test } from 'node:test';
import assert from 'node:assert/strict';
import { UNITS } from '../public/js/content/index.js';
import { W, OPTIMIZERS, simulateOptimizer } from '../public/js/widgets.js';
import { parseShape, rich, rng } from '../public/js/util.js';
import { vary } from '../public/js/vary.js';

const lessons = UNITS.flatMap((u) => u.lessons);
const cards = lessons.flatMap((l) => l.cards ?? []);
const STEP_TYPES = new Set(['learn', 'task', 'mc', 'tf', 'num', 'shape', 'match', 'explain']);
const KINDS = new Set(['recall', 'derive', 'diagnose', 'defend', 'code']);

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

// Each item as written, plus seeded samples of the ones that draw fresh numbers.
const SEEDS = Array.from({ length: 40 }, (_, i) => i + 1);
const instances = (item) => (item.gen ? SEEDS.map((seed) => vary(item, rng(seed))) : [item]);

test('every step is well formed', () => {
  for (const l of lessons) {
    for (const s of l.steps.flatMap(instances)) {
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
      if (s.t === 'explain') {
        assert.ok(KINDS.has(s.kind), `explain step needs a kind in ${where}`);
        assert.ok(s.model && s.rubric?.length >= 3, `explain step needs a model answer and 3+ rubric points in ${where}`);
      }
    }
  }
});

test('every lesson unlocks at least three review cards', () => {
  for (const l of lessons) assert.ok((l.cards ?? []).length >= 3, l.id);
  for (const c of cards.flatMap(instances)) {
    assert.ok(c.front && c.back, c.id);
    if (c.rubric) assert.ok(KINDS.has(c.kind) && c.rubric.length >= 3, `${c.id}: rubric cards need a kind and 3+ points`);
  }
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

test('the optimizer widget reproduces the numbers taught in the weight-decay lesson', () => {
  const end = (id, opts) => simulateOptimizer(id, opts).at(-1);
  assert.deepEqual(OPTIMIZERS.map((o) => end(o.id, { steps: 200 }).toFixed(3)), ['0.819', '0.120', '0.819', '0.016', '0.819']);
  const busy = OPTIMIZERS.map((o) => end(o.id, { steps: 2000, busy: true }));
  assert.deepEqual(busy.map((w) => w.toFixed(3)).slice(0, 4), ['1.818', '1.818', '1.980', '1.818']);
  assert.equal(busy[4].toFixed(2), '1.98');
});

test('the new course has derive, diagnose and defend practice', () => {
  const kinds = new Set([...lessons.flatMap((l) => l.steps), ...cards].filter((x) => x.kind).map((x) => x.kind));
  for (const k of ['derive', 'diagnose', 'defend']) assert.ok(kinds.has(k), k);
});

test('questions and cards with a generator draw different numbers each time', () => {
  const items = [...lessons.flatMap((l) => l.steps), ...cards].filter((x) => x.gen);
  assert.ok(items.length >= 100, `only ${items.length} items vary`);
  for (const item of items) {
    const shown = SEEDS.map((seed) => vary(item, rng(seed)));
    const texts = shown.map((x) => [x.q, x.code, x.fig, x.front].join('|'));
    assert.ok(new Set(texts).size >= 3, `${item.id ?? item.q}: only ${new Set(texts).size} different versions in ${SEEDS.length} draws`);
    for (const x of shown) {
      const text = [x.q, x.code, x.fig, x.ex, x.front, x.back].filter(Boolean).join(' ');
      assert.doesNotMatch(text, /undefined|NaN|Infinity|\[object/, `${item.id ?? item.q}: ${text}`);
      assert.equal(x.t, item.t, 'a generator never changes the step type');
      assert.ok(x.py === undefined || /\bans\s*=/.test(x.py), `${item.id ?? item.q}: py must set ans`);
    }
  }
});

test('every numeric and shape question draws fresh numbers', () => {
  for (const l of lessons) for (const s of l.steps) if (s.t === 'num' || s.t === 'shape') assert.ok(s.gen, `${l.id}: ${s.q}`);
});

test('flashcards are context review: none makes you compute', () => {
  for (const c of cards) assert.ok(!c.gen, `${c.id}: number drills belong in lessons, not on cards`);
});

test('an item without a generator is shown as written', () => {
  const card = cards.find((c) => !c.gen);
  assert.equal(vary(card), card);
});
