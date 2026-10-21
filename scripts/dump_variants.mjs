// Prints seeded samples of every question and card that draws fresh numbers (public/js/vary.js),
// as JSON, for scripts/verify_numpy.py to check against NumPy.
//
//   node scripts/dump_variants.mjs [samples per item]
import { UNITS } from '../public/js/content/index.js';
import { vary } from '../public/js/vary.js';
import { rng } from '../public/js/util.js';

const SAMPLES = Number(process.argv[2] ?? 25);
const out = [];
for (const u of UNITS) {
  for (const l of u.lessons) {
    for (const item of [...l.steps, ...(l.cards ?? [])]) {
      if (!item.gen) continue;
      for (let seed = 1; seed <= SAMPLES; seed++) {
        const x = vary(item, rng(seed * 7919));
        if (x.py === undefined) continue;
        // Cards show answers rounded to 3 decimals; steps carry their own tolerance.
        out.push({ where: `${l.id} ${item.id ?? item.t}`, seed, a: x.a, tol: x.tol ?? item.tol ?? (item.id ? 1e-3 : 0), py: x.py });
      }
    }
  }
}
process.stdout.write(JSON.stringify(out));
