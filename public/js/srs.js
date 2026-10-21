// Spaced-repetition scheduling, modeled on Anki's SM-2 variant.
//
// A card with no state is new. Grading moves it through:
//   learn    -> short steps in minutes (1m, 10m) until it graduates
//   review   -> intervals in days that grow by the card's ease factor
//   relearn  -> after a lapse ("Again" on a review card), a 10-minute step, then back to review
//
// Grades: 1 = Again, 2 = Hard, 3 = Good, 4 = Easy.
// This module is pure (no DOM, no storage) so the browser and the tests share it.

export const MINUTE = 60_000;
export const HOUR = 3_600_000;
export const DAY = 86_400_000;

export const CONFIG = {
  learnSteps: [1, 10],      // minutes
  relearnSteps: [10],       // minutes
  graduatingDays: 1,
  easyDays: 4,
  startEase: 2.5,
  minEase: 1.3,
  hardFactor: 1.2,
  easyBonus: 1.3,
  lapseFactor: 0.5,         // a lapsed card keeps half its interval
  maxDays: 36500,
  rolloverHour: 4,          // a study day starts at 4 a.m., like Anki
};

export const GRADES = ['Again', 'Hard', 'Good', 'Easy'];

// Start of the study day that contains time t, in local time.
export function dayStart(t, cfg = CONFIG) {
  const d = new Date(t - cfg.rolloverHour * HOUR);
  d.setHours(0, 0, 0, 0);
  return d.getTime() + cfg.rolloverHour * HOUR;
}

const roundEase = (e) => Math.round(e * 100) / 100;

function reviewIntervals(ivl, ease, cfg) {
  const hard = Math.max(ivl + 1, Math.round(ivl * cfg.hardFactor));
  const good = Math.max(hard + 1, Math.round(ivl * ease));
  const easy = Math.max(good + 1, Math.round(ivl * ease * cfg.easyBonus));
  return [hard, good, easy].map((d) => Math.min(cfg.maxDays, d));
}

export function schedule(card, grade, now = Date.now(), cfg = CONFIG) {
  if (![1, 2, 3, 4].includes(grade)) throw new Error(`grade must be 1-4, got ${grade}`);
  const c = card
    ? { ...card }
    : { phase: 'learn', step: 0, ivl: 0, ease: cfg.startEase, reps: 0, lapses: 0, due: now, last: 0, intro: now };
  c.reps += 1;
  c.last = now;

  const graduate = (days) => {
    c.phase = 'review';
    c.step = 0;
    c.ivl = Math.min(cfg.maxDays, Math.max(1, days));
    c.due = dayStart(now, cfg) + c.ivl * DAY;
  };

  if (c.phase === 'learn' || c.phase === 'relearn') {
    const steps = c.phase === 'learn' ? cfg.learnSteps : cfg.relearnSteps;
    const step = Math.min(c.step, steps.length - 1);
    if (grade === 1) {
      c.step = 0;
      c.due = now + steps[0] * MINUTE;
    } else if (grade === 2) {
      const minutes = step === 0 && steps.length > 1 ? (steps[0] + steps[1]) / 2 : steps[step] * 1.5;
      c.due = now + minutes * MINUTE;
    } else if (grade === 3) {
      if (step + 1 < steps.length) {
        c.step = step + 1;
        c.due = now + steps[c.step] * MINUTE;
      } else {
        graduate(c.phase === 'learn' ? cfg.graduatingDays : c.ivl);
      }
    } else {
      graduate(c.phase === 'learn' ? cfg.easyDays : c.ivl + 1);
    }
  } else {
    if (grade === 1) {
      c.lapses += 1;
      c.ease = Math.max(cfg.minEase, c.ease - 0.2);
      c.ivl = Math.max(1, Math.round(c.ivl * cfg.lapseFactor));
      c.phase = 'relearn';
      c.step = 0;
      c.due = now + cfg.relearnSteps[0] * MINUTE;
    } else {
      const [hard, good, easy] = reviewIntervals(c.ivl, c.ease, cfg);
      if (grade === 2) c.ease = Math.max(cfg.minEase, c.ease - 0.15);
      if (grade === 4) c.ease += 0.15;
      c.ivl = grade === 2 ? hard : grade === 3 ? good : easy;
      c.due = dayStart(now, cfg) + c.ivl * DAY;
    }
  }
  c.ease = roundEase(c.ease);
  return c;
}

// What each of the four buttons would do, for labeling them ("10m", "4d").
export function preview(card, now = Date.now(), cfg = CONFIG) {
  return [1, 2, 3, 4].map((g) => {
    const next = schedule(card, g, now, cfg);
    return next.phase === 'review' ? { days: next.ivl } : { ms: next.due - now };
  });
}

export function formatInterval({ days, ms }) {
  if (days != null) {
    if (days < 30) return `${days}d`;
    if (days < 365) return `${+(days / 30).toFixed(1)}mo`;
    return `${+(days / 365).toFixed(1)}y`;
  }
  if (ms < MINUTE) return '<1m';
  if (ms < HOUR) return `${Math.round(ms / MINUTE)}m`;
  return `${+(ms / HOUR).toFixed(1)}h`;
}

export const isDue = (card, now = Date.now()) => !!card && card.due <= now;
