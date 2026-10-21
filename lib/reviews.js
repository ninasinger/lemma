// Validation and merging for spaced-repetition card states. The scheduling itself lives in
// public/js/srs.js and runs in the browser; the server only checks and stores the results.

const CARD_ID = /^[a-z0-9-]{1,64}$/;
const PHASES = new Set(['learn', 'review', 'relearn']);
const MAX_CARDS_PER_REQUEST = 2000;
const MAX_TIME = 8.64e15; // the largest valid JS date

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isTime = (t) => Number.isInteger(t) && t >= 0 && t <= MAX_TIME;
const isCount = (n) => Number.isInteger(n) && n >= 0 && n <= 1_000_000;

export function sanitizeCardState(s) {
  if (!isPlainObject(s)) return null;
  const { phase, step, ivl, ease, reps, lapses, due, last, intro } = s;
  if (!PHASES.has(phase)) return null;
  if (!Number.isInteger(step) || step < 0 || step > 20) return null;
  if (typeof ivl !== 'number' || !Number.isFinite(ivl) || ivl < 0 || ivl > 36500) return null;
  if (typeof ease !== 'number' || !Number.isFinite(ease) || ease < 1 || ease > 10) return null;
  if (!isCount(reps) || !isCount(lapses)) return null;
  if (!isTime(due) || !isTime(last) || !isTime(intro)) return null;
  return { phase, step, ivl, ease, reps, lapses, due, last, intro };
}

// Accepts { cards: { [cardId]: state } }. Returns the clean map, or null if anything is off.
export function sanitizeReviewBatch(body) {
  if (!isPlainObject(body) || !isPlainObject(body.cards)) return null;
  const entries = Object.entries(body.cards);
  if (entries.length > MAX_CARDS_PER_REQUEST) return null;
  const clean = {};
  for (const [id, state] of entries) {
    const s = CARD_ID.test(id) ? sanitizeCardState(state) : null;
    if (!s) return null;
    clean[id] = s;
  }
  return clean;
}

// For each card, the state with the most recent review wins. An old tab or an offline phone
// syncing late can't roll a card back to an earlier review.
export function mergeReviews(saved, incoming) {
  const out = { ...saved };
  for (const [id, state] of Object.entries(incoming)) {
    if (!out[id] || state.last >= out[id].last) out[id] = state;
  }
  return out;
}
