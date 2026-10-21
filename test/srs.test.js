import { test } from 'node:test';
import assert from 'node:assert/strict';
import { schedule, preview, formatInterval, dayStart, CONFIG, DAY, MINUTE, HOUR } from '../public/js/srs.js';

// A fixed local time well away from the 4 a.m. rollover.
const NOW = new Date(2026, 9, 21, 14, 0, 0).getTime();

test('a new card steps through 1m and 10m before graduating to 1 day', () => {
  let c = schedule(null, 3, NOW);
  assert.equal(c.phase, 'learn');
  assert.equal(c.due - NOW, 10 * MINUTE);
  assert.equal(c.intro, NOW);
  c = schedule(c, 3, NOW + 10 * MINUTE);
  assert.equal(c.phase, 'review');
  assert.equal(c.ivl, 1);
  assert.equal(c.due, dayStart(NOW) + DAY);
});

test('Again on a new card repeats the first step', () => {
  const c = schedule(null, 1, NOW);
  assert.equal(c.phase, 'learn');
  assert.equal(c.step, 0);
  assert.equal(c.due - NOW, 1 * MINUTE);
});

test('Easy on a new card graduates straight to 4 days', () => {
  const c = schedule(null, 4, NOW);
  assert.equal(c.phase, 'review');
  assert.equal(c.ivl, CONFIG.easyDays);
});

test('review intervals grow by the ease factor and stay ordered Hard < Good < Easy', () => {
  const card = { phase: 'review', step: 0, ivl: 10, ease: 2.5, reps: 5, lapses: 0, due: NOW, last: NOW - 10 * DAY, intro: 0 };
  const [hard, good, easy] = [2, 3, 4].map((g) => schedule(card, g, NOW).ivl);
  assert.equal(good, 25);
  assert.ok(hard < good && good < easy, `${hard} < ${good} < ${easy}`);
  assert.equal(schedule(card, 2, NOW).ease, 2.35);
  assert.equal(schedule(card, 4, NOW).ease, 2.65);
});

test('a lapse halves the interval, lowers ease and sends the card to relearning', () => {
  const card = { phase: 'review', step: 0, ivl: 20, ease: 2.5, reps: 8, lapses: 0, due: NOW, last: 0, intro: 0 };
  let c = schedule(card, 1, NOW);
  assert.deepEqual([c.phase, c.ivl, c.ease, c.lapses], ['relearn', 10, 2.3, 1]);
  assert.equal(c.due - NOW, 10 * MINUTE);
  c = schedule(c, 3, NOW + 10 * MINUTE);
  assert.deepEqual([c.phase, c.ivl], ['review', 10]);
});

test('ease never drops below the minimum', () => {
  let c = { phase: 'review', step: 0, ivl: 5, ease: 1.35, reps: 3, lapses: 0, due: NOW, last: 0, intro: 0 };
  for (let i = 0; i < 5; i++) c = schedule({ ...c, phase: 'review' }, 1, NOW);
  assert.equal(c.ease, CONFIG.minEase);
});

test('day intervals land on the 4 a.m. rollover, not 24 hours after the review', () => {
  const lateNight = new Date(2026, 9, 21, 23, 30).getTime();
  const c = schedule(null, 4, lateNight);
  assert.equal(new Date(c.due).getHours(), CONFIG.rolloverHour);
  const early = new Date(2026, 9, 22, 2, 0).getTime();
  assert.equal(dayStart(early), dayStart(lateNight), '2 a.m. still belongs to the previous study day');
});

test('preview labels the four buttons', () => {
  const labels = preview(null, NOW).map(formatInterval);
  assert.deepEqual(labels, ['1m', '6m', '10m', '4d']);
  assert.equal(formatInterval({ days: 45 }), '1.5mo');
  assert.equal(formatInterval({ ms: 2 * HOUR }), '2h');
});

test('rejects grades outside 1-4', () => {
  assert.throws(() => schedule(null, 5, NOW));
});
