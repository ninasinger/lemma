# Writing lessons and flashcards

Each unit lives in its own file in `public/js/content/` and is listed, in course order, in `public/js/content/index.js`. A unit is plain data:

```js
export default {
  id: 'mx',                 // permanent
  title: 'Matrices',
  sub: 'One-line promise shown on the unit banner.',
  c: 'mx',                  // color token: mx, pca, svd, l2, wd, dr or gold (add new ones in app.css)
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

Any step can also carry `code`, which is shown as a highlighted Python block under the question.

## Rich text

- `{{1 2; 3 4}}` renders a matrix (rows split by `;`). `{{1; 2; 3}}` is a column vector.
- `<span class="m">…</span>` is inline math; `<div class="eq">…</div>` is a display equation.
- `<pre class="py">…</pre>` is a syntax-highlighted Python block. Escape `<` as `&lt;` inside HTML strings.
- Count indices from 0, exactly like NumPy.

## Widgets

`pca`, `svd`, `img`, `ridge`, `contour`, `decay`, `drop`, `scree`, `dot`, `matmul`, `axis`. They live in `public/js/widgets.js`; each takes options (`wo`) and returns `{ solved() }`. `matmul` is the most reusable:

```js
{ w: 'matmul', wo: { A: [[1, 2], [3, 4]], B: [[5], [6]], names: ['W', 'x', 'Wx'],
                     aRows: [...], k: [...], bCols: [...],      // optional axis labels
                     explain: (i, j) => 'what C[i, j] means',   // optional
                     target: [1, 0] } }                         // makes it a task
```

## Flashcards

Every lesson should end with 3–6 cards: `{ id, front, back }`, both HTML. A card is unlocked when its lesson is finished, then scheduled with Anki's SM-2 rules (`public/js/srs.js`). Good cards ask one thing, have a short answer, and include at least one about **shapes** whenever code is involved.

## Checking your work

```bash
npm test             # content structure, ids, widgets, scheduler, server
npm run verify:py    # runs every NumPy claim the lessons make
```

When a lesson teaches a NumPy fact, add a matching `check(...)` to `scripts/verify_numpy.py`.
