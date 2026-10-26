# Writing lessons and flashcards

Each unit lives in its own file in `public/js/content/` and is listed, in course order, in `public/js/content/index.js`. A unit is plain data:

```js
export default {
  id: 'mx',                 // permanent
  title: 'Matrices',
  sub: 'One-line promise shown on the unit banner.',
  c: 'mx',                  // color token: mx, dv, opt, pca, svd, gen, l2, wd, dr or gold (add new ones in app.css)
  guide: `…HTML…`,          // the Guidebook cheat sheet
  lessons: [
    { id: 'mx-1', title: 'Dot products', steps: [ … ], cards: [ … ] },
  ],
};
```

**Ids are permanent.** Lesson ids store which lessons are finished and card ids store review history. Renaming one loses that history. Use lowercase letters, digits and dashes; `npm test` enforces this and checks that ids are unique.

## Step types

| `t` | Fields | Notes |
| --- | --- | --- |
| `learn` | `title`, `body`, optional `w` + `wo` | A concept card, optionally with an interactive widget. |
| `task` | `title`, `body`, `w`, `wo`, `hint`, `ex` | The widget's `solved()` decides correctness. A miss shows the hint and costs no heart. |
| `mc` | `q`, `o`, `ex` | **The first option is the correct one**; options are shuffled on screen. |
| `tf` | `q`, `a` (boolean), `ex` | |
| `num` | `q`, `a`, `tol`, optional `suf` | Accepts answers within `tol`. |
| `shape` | `q`, `code`, `a`, `ex`, `err` | Predict a NumPy shape. `a` is like `'(n, d)'`, `'(3,)'`, `'()'`, or `'error'` (then set `err: true`). |
| `match` | `q`, `p` (pairs), `ex` | Tap-to-match pairs. |
| `explain` | `kind`, `q`, `model`, `rubric`, optional `secs` | Open-ended interview practice. The learner answers (typed or out loud) with a timer, then sees the model answer and ticks the rubric points they covered. Under 60% coverage counts as a miss for accuracy; no hearts are lost. |

Any step can also carry `code`, which is shown as a highlighted Python block under the question.

## Passing a lesson

A lesson passes when every question has been answered right. A missed question (including an interview prompt scored under 60%) comes back at the end of the run with new numbers until it is, so only the misses are ever asked again; running out of hearts ends the run. A run with no misses is flawless. The next unit stays locked until every lesson before it has passed.

## Fresh numbers

Every `num` and `shape` step carries `gen(v)`: a function that returns the fields that change, built from random numbers. The app calls it each time the item is shown (a retry, a missed question coming back, a flashcard review), so the same numbers never come twice. `npm test` fails if a `num` or `shape` step has no `gen`.

```js
const gen = {
  dot(v) {
    const a = v.vec(3), b = v.vec(3), s = dot(a, b);
    return { q: `a = ${list(a)} and b = ${list(b)}. What is a·b?`, a: s, ex: `${dotTerms(a, b)} = ${num(s)}.`,
      py: `ans = np.array(${py(a)}) @ np.array(${py(b)})` };
  },
};
{ t: 'num', q: 'a = [2, −1, 3] and b = [4, 0, 1]. What is a·b?', a: 11, tol: 0, ex: '…', gen: gen.dot }
```

- `v` has `int`, `nz`, `pick`, `sign`, `vec`, `mat` and `sizes(k)` (k different realistic sizes, so a shape mix-up can't hide behind two equal lengths). Format with `num`, `list`, `mtx`, `shape`, `big` and `dotTerms` from `public/js/vary.js`.
- Return `py`: Python that recomputes the answer from the same numbers and sets `ans` (a number, a list, or a shape tuple; a `ValueError` means `error`). `npm run verify:py` runs 25 seeded samples of every generator through NumPy. Write it a different way from the JavaScript where you can, e.g. run the actual NumPy line for a shape.
- Keep the written `q`, `a` and `ex` as the canonical example; `scripts/verify_numpy.py` checks those too.

## Four kinds of practice

Short recall builds memory; interviews test whether you can produce and defend an answer. For each important concept, aim for all four:

| `kind` | Prompt style | Example |
| --- | --- | --- |
| `recall` | State it | “Write the AdamW update.” |
| `derive` | Show it | “Show exactly when L2 and decay are equivalent.” |
| `diagnose` | Explain an observation | “A learning-rate change alters shrinkage. Why?” |
| `defend` | Choose and justify under constraints | “Pick a regularization experiment for a fixed budget.” |

(`code` is reserved for implementation exercises.) Write rubrics as 3–5 observable points (“Gives [0.99, 0.99] for Adam + L2”), not vague ones (“understands Adam”). Teach train/validation patterns as **clues under stated assumptions**, never as verdicts.

## Rich text

- `{{1 2; 3 4}}` renders a matrix (rows split by `;`). `{{1; 2; 3}}` is a column vector.
- `<span class="m">…</span>` is inline math; `<div class="eq">…</div>` is a display equation.
- `<pre class="py">…</pre>` is a syntax-highlighted Python block. Escape `<` as `&lt;` inside HTML strings.
- Count indices from 0, exactly like NumPy.

## Widgets

`pca`, `svd`, `img`, `ridge`, `contour`, `decay`, `drop`, `scree`, `dot`, `matmul`, `axis`, `biasvar`, `curves`, `optim`. They live in `public/js/widgets.js`; each takes options (`wo`) and returns `{ solved() }`. `matmul` is the most reusable:

```js
{ w: 'matmul', wo: { A: [[1, 2], [3, 4]], B: [[5], [6]], names: ['W', 'x', 'Wx'],
                     aRows: [...], k: [...], bCols: [...],      // optional axis labels
                     explain: (i, j) => 'what C[i, j] means',   // optional
                     target: [1, 0] } }                         // makes it a task
```

## Flashcards

Every lesson should end with 3–6 cards: `{ id, front, back }`, both HTML. Cards are **context review**: concepts, rules, shapes and judgments, never arithmetic or derivations. Number drills belong in the lesson, where they get fresh numbers; `npm test` fails if a card has a `gen`. A card is unlocked when its lesson is finished, then scheduled with Anki's SM-2 rules (`public/js/srs.js`). Good cards ask one thing, have a short answer, and include at least one about **shapes** whenever code is involved.

A card can also be an interview card: add `kind`, `rubric` (3–5 points) and optionally `secs`. In review it shows an answer box and a timer, and after the reveal the rubric; the share of points ticked suggests a grade (under 50% Again, under 80% Hard, otherwise Good).

## Checking your work

```bash
npm test             # content structure, ids, widgets, scheduler, server
npm run verify:py    # runs every NumPy claim the lessons make
```

When a lesson teaches a NumPy fact, add a matching `check(...)` to `scripts/verify_numpy.py`.
