// Fresh numbers for practice questions and flashcards.
//
// A step or a card can carry gen(v). It returns the fields that change (q, code, a and ex for a
// step; front and back for a card), built with the random helpers in v. The app calls it every
// time the item is shown, so a retry, a question that comes back after a miss, or a flashcard
// review never repeats the same numbers. Everything else on the item (type, tolerance, id) stays.
//
// gen also returns py: Python that recomputes the answer from the same numbers and stores it in
// `ans` (a number, a list, or a shape tuple). It is never shown. scripts/verify_numpy.py runs a
// sample of every generator through it, so a generator can't quietly teach a wrong answer.
import { shuffle } from './util.js';

// Sizes that look like real data: a few you can picture, and typical feature and layer widths.
const SIZES = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128, 200, 256, 500, 784, 1000];

export function helpers(r = Math.random) {
  const int = (lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
  const pick = (a) => a[Math.floor(r() * a.length)];
  return {
    int,
    pick,
    // A non-zero integer in [lo, hi].
    nz(lo, hi) {
      let x;
      do x = int(lo, hi); while (x === 0);
      return x;
    },
    sign: () => (r() < 0.5 ? -1 : 1),
    vec: (n, lo = -4, hi = 5) => Array.from({ length: n }, () => int(lo, hi)),
    mat: (m, n, lo = -3, hi = 5) => Array.from({ length: m }, () => Array.from({ length: n }, () => int(lo, hi))),
    // k different sizes, so a shape question never hides a mix-up behind two equal lengths.
    sizes: (k, from = SIZES) => shuffle(from, r).slice(0, k),
  };
}

// Numbers the way the lessons print them: a real minus sign and no floating-point noise.
export const num = (x, digits = 6) => String(+(+x).toFixed(digits)).replace('-', '−');
// A factor inside a product, in parentheses when negative: (−1)·4.
export const term = (x) => (x < 0 ? `(${num(x)})` : num(x));
// Large counts with thousands separators: 2,500,000.
export const big = (x) => x.toLocaleString('en-US');
export const list = (v) => `[${v.map(num).join(', ')}]`;
// A matrix in the lessons' rich-text form, e.g. {{1 2; 3 4}}.
export const mtx = (M) => `{{${M.map((row) => row.map(num).join(' ')).join('; ')}}}`;
export const shape = (...d) => (d.length === 1 ? `(${d[0]},)` : `(${d.join(', ')})`);
// Σ x[k]·y[k] written out, e.g. "2·4 + (−1)·0 + 3·1".
export const dotTerms = (x, y) => x.map((v, k) => `${term(v)}·${term(y[k])}`).join(' + ');
export const dot = (x, y) => x.reduce((s, v, k) => s + v * y[k], 0);
// A Python literal for a number or a nested list.
export const py = (x) => JSON.stringify(x);

// The item as it should be shown this time.
export const vary = (item, r = Math.random) => (item.gen ? { ...item, ...item.gen(helpers(r)) } : item);
