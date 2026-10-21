const LESSON_ID = /^[a-z0-9-]{1,40}$/;
const DAY = /^\d{4}-\d{2}-\d{2}$/;
const MAX_LESSONS = 200;

export const emptyProgress = () => ({ done: {}, xp: 0, streak: 0, last: null, muted: false });

const isCount = (n) => Number.isInteger(n) && n >= 0 && n <= 10_000_000;
const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

// Returns a clean copy of a progress object, or null if the input isn't shaped like one.
export function sanitizeProgress(input) {
  if (!isPlainObject(input)) return null;
  const { done = {}, xp = 0, streak = 0, last = null, muted = false } = input;
  if (!isPlainObject(done)) return null;
  const ids = Object.keys(done).filter((id) => done[id] === true);
  if (ids.length > MAX_LESSONS || !ids.every((id) => LESSON_ID.test(id))) return null;
  if (!isCount(xp) || !isCount(streak)) return null;
  if (last !== null && !(typeof last === 'string' && DAY.test(last))) return null;
  if (typeof muted !== 'boolean') return null;
  return { done: Object.fromEntries(ids.map((id) => [id, true])), xp, streak, last, muted };
}

// Combines the saved progress with what a browser sends, so a stale tab or a second
// device can never erase work: finished lessons are unioned, XP keeps the higher total,
// and the streak comes from whichever side played most recently.
export function mergeProgress(saved, incoming) {
  const savedDay = saved.last ?? '';
  const incomingDay = incoming.last ?? '';
  let recent;
  if (incomingDay !== savedDay) recent = incomingDay > savedDay ? incoming : saved;
  else recent = incoming.streak >= saved.streak ? incoming : saved;
  return {
    done: { ...saved.done, ...incoming.done },
    xp: Math.max(saved.xp, incoming.xp),
    streak: recent.streak,
    last: recent.last,
    muted: incoming.muted,
  };
}
