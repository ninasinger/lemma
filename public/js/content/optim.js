// Unit: optimization I. Convexity, gradient descent and SGD, batch size, why SGD works on
// non-convex networks, and a bias/variance review. Train/validation patterns are clues under
// stated assumptions (same measurement, representative validation), not verdicts.
import { num, term, list, big, py } from '../vary.js';

// Fresh numbers every time a question or card is shown (see ../vary.js). py is checked by
// scripts/verify_numpy.py, with deriv() its complex-step derivative.
const coef = (a) => (a === 1 ? '' : num(a));
// a(w − c)², written w² when c = 0 and (w + 3)² when c is negative.
const sq = (a, c) => (c === 0 ? `${coef(a)}w²` : `${coef(a)}(w ${c < 0 ? '+' : '−'} ${Math.abs(c)})²`);
const grad = (a, c) => (c === 0 ? `${num(2 * a)}w` : `${num(2 * a)}(w ${c < 0 ? '+' : '−'} ${Math.abs(c)})`);
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;

const LABEL = { var: 'High variance (overfitting)', bias: 'High bias (underfitting)', ok: 'Healthy: a small, expected gap' };
const FIXES = [
  ['Add weight decay', 'var'], ['Add dropout', 'var'], ['Stop training earlier', 'var'], ['Get more training data', 'var'],
  ['Use data augmentation', 'var'], ['Use a smaller model', 'var'], ['Make the model bigger', 'bias'],
  ['Train longer (the loss is still falling)', 'bias'], ['Reduce weight decay', 'bias'], ['Improve the features or architecture', 'bias'],
];

const gen = {
  quadMin(v) {
    const a = v.pick([1, 2, 3, 0.5]), c = v.nz(-5, 6), d = v.int(1, 9), where = v.int(0, 1) === 1;
    return where
      ? { q: `L(w) = ${sq(a, c)} + ${d}. This is convex. Where is its minimum?`, a: c,
        ex: `∇L = ${grad(a, c)}, which is 0 only at w = ${num(c)}. For a convex function a zero gradient means a global minimum.`,
        py: `ws = np.linspace(-10, 10, 200001)\nans = ws[np.argmin(${a} * (ws - ${c})**2 + ${d})]` }
      : { q: `L(w) = ${sq(a, c)} + ${d}. This is convex. What is the smallest value L can take?`, a: d,
        ex: `At w = ${num(c)} the squared term is 0, leaving ${d}. Convex, so that local minimum is the global one.`,
        py: `ws = np.linspace(-10, 10, 200001)\nans = (${a} * (ws - ${c})**2 + ${d}).min()` };
  },
  direction(v) {
    const g = v.nz(-9, 9), down = g > 0;
    return { q: `At the current w, ∇L(w) = ${num(g)}. Which way does one gradient-descent step move w?`,
      o: [down ? 'Down: w decreases' : 'Up: w increases', down ? 'Up: w increases' : 'Down: w decreases', 'It stays put', 'It depends on the sign of η'],
      ex: `The gradient points uphill, so the step goes the other way: w − η × ${term(g)} ${down ? 'lowers' : 'raises'} w.` };
  },
  gdGiven(v) {
    const w = v.pick([-2, -1, 0.5, 1, 2, 3]), g = v.nz(-8, 8), eta = v.pick([0.1, 0.05, 0.5, 0.01]), w1 = +(w - eta * g).toFixed(6);
    return { q: `w = ${num(w)}, ∇L(w) = ${num(g)}, η = ${num(eta)}. What is w after one gradient-descent step?`, a: w1,
      ex: `w − η∇L = ${num(w)} − ${num(eta)} × ${term(g)} = ${num(w1)}. The gradient is ${g > 0 ? 'positive' : 'negative'}, so w moves ${g > 0 ? 'down' : 'up'}: downhill.`,
      py: `ans = ${w} - ${eta} * ${g}` };
  },
  gdStep(v) {
    const a = v.pick([1, 2, 0.5]), c = v.int(-3, 3), w0 = c + v.nz(-4, 4), eta = v.pick([0.1, 0.05, 0.25]);
    const g = 2 * a * (w0 - c), w1 = +(w0 - eta * g).toFixed(6);
    return { q: `L(w) = ${sq(a, c)}, w = ${num(w0)}, η = ${num(eta)}. What is w after one gradient-descent step?`, a: w1,
      ex: `∇L = ${grad(a, c)} = ${num(g)} at w = ${num(w0)}. Step: ${num(w0)} − ${num(eta)} × ${term(g)} = ${num(w1)}, closer to the minimum at ${num(c)}.`,
      py: `ans = ${w0} - ${eta} * deriv(lambda w: ${a} * (w - ${c})**2, ${w0})` };
  },
  fullGrad(v) {
    const gs = v.vec(4, -6, 9), m = mean(gs);
    return { q: `Four training examples have per-example gradients ${list(gs)}. What is the full-batch gradient, their mean?`, a: m,
      ex: `(${gs.map(term).join(' + ')}) / 4 = ${num(m)}.`, py: `ans = np.mean(${py(gs)})` };
  },
  batchGrad(v) {
    const gs = v.vec(4, -6, 9), i = v.int(0, 2), j = v.int(i + 1, 3), m = (gs[i] + gs[j]) / 2;
    return { q: `Per-example gradients are ${list(gs)}. A mini-batch draws examples ${i} and ${j} (counting from 0). What is the mini-batch gradient?`, a: m,
      ex: `(${num(gs[i])} + ${term(gs[j])}) / 2 = ${num(m)}. The full gradient is ${num(mean(gs))}: one batch can point somewhat away from it.`,
      py: `g = np.array(${py(gs)})\nans = g[[${i}, ${j}]].mean()` };
  },
  allBatches(v) {
    const gs = v.vec(4, -6, 9), m = mean(gs);
    return { q: `Per-example gradients are ${list(gs)}. Average the gradients of all 6 possible 2-example mini-batches. What do you get?`, a: m,
      ex: `Each example appears in 3 of the 6 batches, so the average is exactly the full gradient, ${num(m)}. That is what “unbiased” means: right on average.`,
      py: `import itertools\nans = np.mean([np.mean(b) for b in itertools.combinations(${py(gs)}, 2)])` };
  },
  updates(v) {
    const N = v.pick([32000, 50000, 60000, 100000, 25600, 12800]), B = v.pick([32, 64, 128, 256, 512, 1024]), u = N / B;
    return { q: `A dataset has ${big(N)} examples and the batch size is ${B}. About how many updates happen per epoch?`, a: +u.toFixed(4),
      ex: Number.isInteger(u) ? `${big(N)} / ${B} = ${big(u)} updates.` : `${big(N)} / ${B} = ${num(u, 2)}: ${Math.floor(u)} full batches, plus one partial batch if it’s kept.`,
      py: `ans = len(range(0, ${N}, ${B}))` };
  },
  fewer(v) {
    const b1 = v.pick([16, 32, 64]), f = v.pick([2, 4, 8, 16, 32]), b2 = b1 * f;
    return { q: `The batch size goes from ${b1} to ${big(b2)}, for the same number of epochs. How many times fewer updates are there?`, a: f,
      ex: `Updates per epoch ≈ N / batch size, so ${big(b2)} / ${b1} = ${f} times fewer, each one cleaner and more expensive.`,
      py: `N = 2**22\nans = (N / ${b1}) / (N / ${b2})` };
  },
  noise(v) {
    const s = v.pick([1, 2, 3, 4, 0.5]), B = v.pick([4, 16, 25, 64]), a = s / Math.sqrt(B);
    return { q: `One coordinate of a single example’s gradient has standard deviation ${num(s)}. What is the standard deviation of a batch-of-${B} average?`, a,
      ex: `σ / √B = ${num(s)} / ${Math.sqrt(B)} = ${num(a, 3)}. Bigger batches, less noise: 4× the batch halves it.`,
      py: `ans = rng.normal(0, ${s}, size=(40000, ${B})).mean(axis=1).std()` };
  },
  saddleValue(v) {
    const t = v.pick([0.5, 1, 1.5, 2, 3]), alongY = v.int(0, 1) === 1, x = alongY ? 0 : t, y = alongY ? t : 0, L = x * x - y * y;
    return { q: `L(x, y) = x² − y² has ∇L = 0 at (0, 0), where L = 0. What is L at (${num(x)}, ${num(y)})?`, a: L,
      ex: alongY ? `0² − ${num(t)}² = ${num(L)}: lower than at (0, 0). Along y the loss goes downhill, even though the slope at the saddle is zero.`
        : `${num(t)}² − 0² = ${num(L)}: higher. Along x the saddle looks like a minimum; along y it doesn’t. That mix is what makes it a saddle.`,
      py: `L = lambda x, y: x**2 - y**2\nans = L(${x}, ${y})` };
  },
  saddleGrad(v) {
    const p = v.nz(-3, 3), q = v.nz(-3, 3), wrtY = v.int(0, 1) === 1;
    return { q: `L(x, y) = x² − y². What is ∂L/∂${wrtY ? 'y' : 'x'} at (${num(p)}, ${num(q)})?`, a: wrtY ? -2 * q : 2 * p,
      ex: wrtY ? `∂L/∂y = −2y = −2 × ${term(q)} = ${num(-2 * q)}.` : `∂L/∂x = 2x = 2 × ${term(p)} = ${num(2 * p)}.`,
      py: `ans = deriv(lambda x, y: x**2 - y**2, ${p}, ${q}, wrt=${wrtY ? 1 : 0})` };
  },
  diagnose(v) {
    const kind = v.pick(['var', 'bias', 'ok']);
    const tr = kind === 'var' ? v.int(1, 3) : kind === 'bias' ? v.int(18, 30) : v.int(2, 4);
    const va = tr + (kind === 'var' ? v.int(10, 20) : kind === 'bias' ? v.int(1, 4) : v.int(1, 2));
    const others = Object.keys(LABEL).filter((k) => k !== kind).map((k) => LABEL[k]);
    const ex = {
      var: 'Low training error, much higher validation error: the model fits its training data but doesn’t generalize. Make memorization harder: weight decay, dropout, early stopping, a smaller model, more data, augmentation.',
      bias: 'Both high and close: the model can’t even fit its training set. Make fitting easier: more capacity, train longer, less regularization, better features. (A failed optimization looks the same; the tiny-batch overfit test tells them apart.)',
      ok: 'Some gap is expected. Both errors are low and the gap is small: healthy.',
    }[kind];
    return { q: `Training error ${tr}%, validation error ${va}%, both measured the same way. What does this most likely suggest?`, o: [LABEL[kind], ...others, 'Nothing: you can’t read anything from errors'], ex };
  },
  fixFor(v) {
    const [fix, kind] = v.pick(FIXES);
    return { q: `“${fix}.” Which problem does that address?`, o: [LABEL[kind], LABEL[kind === 'var' ? 'bias' : 'var'], 'Neither: it only changes training speed'],
      ex: kind === 'var' ? 'It makes memorization harder: the variance side.' : 'It makes fitting easier: the bias side.' };
  },

  cGd(v) {
    const w = v.pick([-1, 1, 2, 3]), g = v.nz(-6, 6), eta = v.pick([0.1, 0.5]), w1 = +(w - eta * g).toFixed(6);
    return { front: `w = ${num(w)}, ∇L = ${num(g)}, η = ${num(eta)}. New w?`, back: `${num(w)} − ${num(eta)} × ${term(g)} = <b>${num(w1)}</b>.`, a: w1, py: `ans = ${w} - ${eta} * ${g}` };
  },
  cBatch(v) {
    const gs = v.vec(4, -6, 9), i = v.int(0, 2), j = v.int(i + 1, 3), m = (gs[i] + gs[j]) / 2;
    return { front: `Gradients ${list(gs)}. Batch of examples ${i} and ${j}: its gradient? The full gradient?`, back: `Batch: <b>${num(m)}</b>. Full: <b>${num(mean(gs))}</b>. Noisy, right on average.`,
      a: [m, mean(gs)], py: `g = np.array(${py(gs)})\nans = [g[[${i}, ${j}]].mean(), g.mean()]` };
  },
  cUpdates(v) {
    const N = v.pick([32000, 64000, 128000]), B = v.pick([32, 64, 128, 256]);
    return { front: `${big(N)} examples, batch size ${B}. Updates per epoch?`, back: `${big(N)} / ${B} = <b>${big(N / B)}</b>.`, a: N / B, py: `ans = len(range(0, ${N}, ${B}))` };
  },
  cDiagnose(v) {
    const D = gen.diagnose(v);
    return { front: D.q.replace(', both measured the same way. What does this most likely suggest?', '. Diagnosis?'), back: `<b>${D.o[0]}</b>. ${D.ex}` };
  },
};

export default {
  id: 'opt',
  title: 'Optimization',
  sub: 'Convexity, SGD and generalization: why noisy steps still find good models.',
  c: 'opt',
  guide: `<h3>One-line memory set</h3><div class="eq hook">Convex: stationary → global min.</div><div class="eq hook">SGD: noisy gradient, right on average.</div><div class="eq hook">Bigger batch → fewer, cleaner updates.</div><div class="eq hook">Bias: make fitting easier. Variance: make memorization harder.</div>
  <h3>Convex vs non-convex</h3><ul><li>Convex (bowl-shaped): every local minimum is a global minimum. Several global minima are possible if they share the value; strictly convex means exactly one.</li><li>Differentiable convex: ∇L(w) = 0 ⇒ global minimum.</li><li>Non-convex: ∇L = 0 only means stationary: local minimum, local maximum or saddle point.</li><li>Neural networks are non-convex: parameters interact through compositions of nonlinear transformations.</li></ul>
  <h3>Gradient descent and SGD</h3><div class="eq">w ← w − η∇L(w)</div><ul><li>The gradient points uphill; the negative gradient points downhill.</li><li>Full batch: the whole training set per update. Mini-batch SGD: a random subset per update, so each update is cheaper and there are many more of them.</li><li>A random mini-batch gradient is unbiased, E[g_batch] = ∇L: noisy direction, right direction on average.</li></ul>
  <h3>Batch size</h3><ul><li>Updates per epoch ≈ N / batch size. 32,000 examples: batch 32 → 1,000; batch 1,024 → about 31.</li><li>Bigger batch: less noise (σ/√B), costlier updates, fewer of them. Larger learning rates often work with larger batches: a heuristic, not a law.</li></ul>
  <h3>Why SGD works on neural networks</h3><ul><li>Mini-batches give useful local descent directions; no full map of the landscape is needed.</li><li>Noise helps: it pushes off saddle points and out of flat regions.</li><li>Large networks have many good solutions; we don’t need the unique global optimum.</li><li>The goal is generalization, not the lowest training loss.</li></ul>
  <h3>Saddle points</h3><ul><li>∇L = 0, but downhill still exists in some directions (L = x² − y² at the origin).</li><li>High-dimensional networks have many of them; SGD noise nudges the parameters off.</li></ul>
  <h3>Batches: variance vs bias</h3><ul><li>A weird random batch is variance: it averages out.</li><li>Systematically weird batching (sorted data, never shuffled) is bias.</li><li>Shuffle so the dataset’s order can’t keep producing correlated or unrepresentative batches.</li></ul>
  <h3>Bias vs variance (same measurement, representative validation)</h3><ul><li>Low train + low-ish val: healthy. Some gap is expected.</li><li>Low train + high val: high variance. Weight decay, dropout, early stopping, smaller model, more data, augmentation.</li><li>High train + high val: high bias. More capacity, train longer, less regularization, better architecture or features.</li></ul>`,
  lessons: [
    {
      id: 'opt-1',
      title: 'Convex vs non-convex',
      steps: [
        { t: 'learn', title: 'Bowls have one kind of bottom', body: '<p>A <b>convex</b> loss is bowl-shaped. The key property: <b>every local minimum is also a global minimum.</b></p><p>A convex function can have several global minima if they share the same value (a flat bottom). A <b>strictly</b> convex function has exactly one.</p><div class="eq">convex: ∇L(w) = 0 ⇒ global min</div>' },
        { t: 'mc', q: 'What does convexity guarantee about local minima?', o: ['Every local minimum is a global minimum', 'There is exactly one minimum', 'Gradient descent converges in one step', 'The minimum value is 0'], ex: 'Exactly one minimum needs strict convexity. A convex function can have a flat bottom of minimizers, all with the same value.' },
        { t: 'num', q: 'L(w) = 2(w − 3)² + 1. This is convex. Where is its minimum?', a: 3, tol: 0, ex: '∇L = 4(w − 3), which is 0 only at w = 3. For a convex function a zero gradient means a global minimum.', gen: gen.quadMin },
        { t: 'tf', q: 'A convex function can have more than one global minimum.', a: true, ex: 'A flat-bottomed bowl such as L(w) = max(0, |w| − 1) is minimized by every w in [−1, 1], all with the same value. Strict convexity rules that out.' },
        { t: 'learn', title: 'Non-convex: zero slope means less', body: '<p>For a non-convex function, ∇L = 0 only says the point is <b>stationary</b>. It could be a local minimum, a local maximum or a <b>saddle point</b>.</p><p>Neural-network losses are non-convex: the parameters interact through compositions of nonlinear transformations, like W₂·relu(W₁x).</p>' },
        { t: 'mc', q: 'For a non-convex loss, ∇L(w) = 0 means w is…', o: ['A stationary point: a local minimum, local maximum or saddle', 'A global minimum', 'A local minimum', 'A bad starting point'], ex: 'Without convexity, a zero gradient only rules out a slope. The point could still be a maximum or a saddle.' },
        { t: 'mc', q: 'L(w) = w³ − 3w has ∇L = 0 at w = −1 and w = 1. What are they?', o: ['w = 1 is a local minimum and w = −1 a local maximum', 'Both are global minima', 'Both are saddle points', 'w = −1 is the global minimum'], ex: 'L(1) = −2 and L(−1) = 2. The curve rises on both sides of 1 and falls on both sides of −1. Neither is global: L keeps falling as w → −∞.' },
        { t: 'num', q: 'L(w) = (w + 2)² + 5. This is convex. What is the smallest value L can take?', a: 5, tol: 0, ex: 'At w = −2 the squared term is 0, leaving 5. Convex, so that local minimum is the global one.', gen: gen.quadMin },
        { t: 'mc', q: 'Why are neural-network losses non-convex?', o: ['Parameters interact through compositions of nonlinear functions', 'They use too much data', 'Their gradients are noisy', 'They have too few parameters'], ex: 'Even a two-layer net, W₂·relu(W₁x), multiplies parameters together through a nonlinearity. Swapping two hidden units gives the same loss, yet averaging the swapped and unswapped networks usually gives a worse one. On a convex loss that can’t happen: the midpoint of two minima is always a minimum too.' },
      ],
      cards: [
        { id: 'opt-convex-local', front: 'What does convexity guarantee about local minima?', back: 'Every local minimum is global.' },
        { id: 'opt-convex-grad', front: 'Differentiable convex L: what does ∇L = 0 imply?', back: 'A global minimum.' },
        { id: 'opt-nonconvex-grad', front: 'Non-convex L: what can ∇L = 0 mean?', back: 'Local minimum, local maximum, or saddle point.' },
        { id: 'opt-strict', front: 'Convex vs strictly convex minima?', back: 'Convex: possibly several, all equal. Strictly convex: exactly one.' },
        { id: 'opt-nn-nonconvex', front: 'Why are neural-network losses non-convex?', back: 'Parameters interact through compositions of nonlinear transformations.' },
      ],
    },
    {
      id: 'opt-2',
      title: 'Gradient descent and SGD',
      steps: [
        { t: 'learn', title: 'Step downhill', body: '<div class="eq">w ← w − η∇L(w)</div><p>η is the <b>learning rate</b>. The gradient points uphill, so subtracting it steps downhill.</p><div class="eq hook">Gradient points uphill; negative gradient points downhill.</div>' },
        { t: 'mc', q: 'At the current w, ∇L(w) = 4. Which way does one gradient-descent step move w?', o: ['Down: w decreases', 'Up: w increases', 'It stays put', 'It depends on the sign of η'], ex: 'The gradient points uphill, so the step goes the other way: w − η × 4 lowers w.', gen: gen.direction },
        { t: 'num', q: 'w = 2, ∇L(w) = 4, η = 0.1. What is w after one gradient-descent step?', a: 1.6, tol: 0.001, ex: 'w − η∇L = 2 − 0.1 × 4 = 1.6. The gradient is positive, so w moves down: downhill.', gen: gen.gdGiven },
        { t: 'num', q: 'L(w) = (w − 3)², w = 5, η = 0.1. What is w after one gradient-descent step?', a: 4.6, tol: 0.001, ex: '∇L = 2(w − 3) = 4 at w = 5. Step: 5 − 0.1 × 4 = 4.6, closer to the minimum at 3.', gen: gen.gdStep },
        { t: 'learn', title: 'Full batch vs mini-batch', body: '<p><b>Full-batch gradient descent</b> computes the gradient on the whole training set before each update.</p><p><b>Mini-batch SGD</b> estimates it from a randomly selected subset. Each update is much cheaper, so training makes many more updates.</p>' },
        { t: 'mc', q: 'What does mini-batch SGD use for each update?', o: ['The gradient on a random subset of training examples', 'The gradient on the whole training set', 'The gradient of one fixed example', 'The validation loss'], ex: 'A random subset gives a cheap estimate of the full gradient, so you can afford many more updates.' },
        { t: 'match', q: 'Match each term to its meaning.', p: [['Full-batch GD', 'Whole training set per update'], ['Mini-batch SGD', 'Random subset per update'], ['η', 'Learning rate: the step size'], ['−∇L', 'The downhill direction']], ex: 'Same update rule; the difference is how much data estimates the gradient.' },
        { t: 'num', q: 'w = −1, ∇L(w) = −6, η = 0.05. What is w after one gradient-descent step?', a: -0.7, tol: 0.001, ex: 'w − η∇L = −1 − 0.05 × (−6) = −0.7. The gradient is negative, so w moves up: downhill.', gen: gen.gdGiven },
      ],
      cards: [
        { id: 'opt-gd-update', front: 'The gradient-descent update?', back: 'w ← w − η∇L(w).' },
        { id: 'opt-uphill', front: 'Which way does the gradient point?', back: 'Uphill. Step along the negative gradient.' },
        { id: 'opt-minibatch', front: 'Full-batch GD vs mini-batch SGD?', back: 'Whole training set per update vs a random subset: cheaper updates, many more of them.' },
        { id: 'opt-gd-drill', front: 'w = 2, ∇L = 4, η = 0.1. New w?', back: '2 − 0.1 × 4 = <b>1.6</b>.', gen: gen.cGd },
      ],
    },
    {
      id: 'opt-3',
      title: 'Noisy but unbiased',
      steps: [
        { t: 'learn', title: 'Right on average', body: '<p>A randomly sampled mini-batch gives an <b>unbiased</b> estimate of the full gradient:</p><div class="eq">E[g_batch] = ∇L</div><p>One batch may point somewhat away from the full-dataset gradient, but averaged over random batches the direction matches.</p><div class="eq hook">SGD = noisy direction, right direction on average.</div>' },
        { t: 'num', q: 'Four training examples have per-example gradients [4, −2, 6, 0]. What is the full-batch gradient, their mean?', a: 2, tol: 0.001, ex: '(4 + (−2) + 6 + 0) / 4 = 2.', gen: gen.fullGrad },
        { t: 'num', q: 'Per-example gradients are [4, −2, 6, 0]. A mini-batch draws examples 0 and 2 (counting from 0). What is the mini-batch gradient?', a: 5, tol: 0.001, ex: '(4 + 6) / 2 = 5. The full gradient is 2: one batch can point somewhat away from it.', gen: gen.batchGrad },
        { t: 'num', q: 'Per-example gradients are [4, −2, 6, 0]. Average the gradients of all 6 possible 2-example mini-batches. What do you get?', a: 2, tol: 0.001, ex: 'Each example appears in 3 of the 6 batches, so the average is exactly the full gradient, 2. That is what “unbiased” means: right on average.', gen: gen.allBatches },
        { t: 'mc', q: 'Properly sampled mini-batch gradients are…', o: ['Unbiased: right on average, noisy one at a time', 'Exact: always equal to the full gradient', 'Biased toward the most recent batch', 'Useless without full-batch steps'], ex: 'Their expected value is the full gradient; individual batches scatter around it.' },
        { t: 'learn', title: 'A weird batch vs weird batching', body: '<p>One strange mini-batch (say, nearly all one class) can give a poor update. If batches are sampled properly, that is <b>variance</b>: it averages out.</p><p>If the sampling itself is systematically unrepresentative, every batch leans the same way: that is <b>bias</b>.</p><div class="eq hook">Weird batch = variance. Systematically weird batching = bias.</div><p><b>Shuffling</b> keeps the dataset’s original order from repeatedly producing correlated or unrepresentative batches.</p>' },
        { t: 'mc', q: 'One randomly drawn batch happens to be almost all cats. Bias or variance?', o: ['Variance: random batches are sometimes odd, and it averages out', 'Bias: every update is now wrong', 'Neither', 'Both'], ex: 'With proper random sampling, an odd batch is just one noisy draw. The next batches pull the average back.' },
        { t: 'mc', q: 'The data is sorted by class and never shuffled, so batches come one class at a time in a fixed order. Bias or variance?', o: ['Bias: the sampling itself is systematically unrepresentative', 'Variance: it averages out', 'Neither', 'It only affects speed'], ex: 'The batches aren’t random draws from the whole dataset, so the updates lean the same way each epoch. Shuffle.' },
        { t: 'mc', q: 'Why shuffle the training data each epoch?', o: ['So the dataset’s order can’t keep producing correlated or unrepresentative batches', 'To make every batch identical', 'To speed up the GPU', 'To change the labels'], ex: 'Shuffling turns each batch back into a random sample, which is what makes the gradient estimate unbiased.' },
      ],
      cards: [
        { id: 'opt-unbiased', front: 'Why can SGD work even though mini-batch gradients are noisy?', back: 'Properly sampled, they are unbiased estimates of the full gradient: the expected direction is right.' },
        { id: 'opt-sgd-hook', front: 'SGD in one line?', back: 'Noisy direction, right direction on average.' },
        { id: 'opt-weird-batch', front: 'A weird batch vs systematically weird batching?', back: 'Variance vs bias.' },
        { id: 'opt-shuffle', front: 'Why shuffle?', back: 'So the data’s order doesn’t keep producing correlated or unrepresentative batches.' },
        { id: 'opt-batch-drill', front: 'Gradients [4, −2, 6, 0]. Batch of examples 0 and 2: its gradient? The full gradient?', back: 'Batch: <b>5</b>. Full: <b>2</b>. Noisy, right on average.', gen: gen.cBatch },
      ],
    },
    {
      id: 'opt-4',
      title: 'Batch size',
      steps: [
        { t: 'learn', title: 'Fewer, cleaner updates', body: '<div class="eq">updates/epoch ≈ N / B</div><p>N is the number of examples and B the batch size. With 32,000 examples: batch 32 gives 1,000 updates per epoch; batch 1,024 gives about 31.</p><p>As the batch grows: gradient estimates get less noisy, each update costs more, there are fewer updates per epoch, and larger learning rates often work. That last one is a heuristic, not a law.</p>' },
        { t: 'num', q: 'A dataset has 32,000 examples and the batch size is 32. About how many updates happen per epoch?', a: 1000, tol: 1, ex: '32,000 / 32 = 1,000 updates.', gen: gen.updates },
        { t: 'num', q: 'A dataset has 32,000 examples and the batch size is 1024. About how many updates happen per epoch?', a: 31.25, tol: 1, ex: '32,000 / 1,024 = 31.25: 31 full batches, plus one partial batch if it’s kept.', gen: gen.updates },
        { t: 'num', q: 'The batch size goes from 32 to 256, for the same number of epochs. How many times fewer updates are there?', a: 8, tol: 0, ex: 'Updates per epoch ≈ N / batch size, so 256 / 32 = 8 times fewer, each one cleaner and more expensive.', gen: gen.fewer },
        { t: 'mc', q: 'As the batch size increases, gradient noise…', o: ['Decreases', 'Increases', 'Stays the same', 'Becomes biased'], ex: 'Averaging more examples shrinks the variance of the estimate, roughly like σ²/B.' },
        { t: 'num', q: 'One coordinate of a single example’s gradient has standard deviation 2. What is the standard deviation of a batch-of-16 average?', a: 0.5, tol: 0.02, ex: 'σ / √B = 2 / 4 = 0.5. Bigger batches, less noise: 4× the batch halves it.', gen: gen.noise },
        { t: 'mc', q: 'Why can larger batches often use larger learning rates?', o: ['The gradient estimate is cleaner, so bigger steps are safer; a heuristic, not a law', 'The learning rate must always scale exactly with batch size', 'The loss becomes convex', 'With fewer updates the learning rate stops mattering'], ex: 'Less noise means a big step is less likely to go the wrong way. It often helps, but it can break down, so check validation.' },
        { t: 'match', q: 'Match each side of the tradeoff.', p: [['Bigger batch', 'Fewer, cleaner, costlier updates'], ['Smaller batch', 'More, noisier, cheaper updates'], ['N / batch size', 'Updates per epoch'], ['Scaling LR with batch', 'A heuristic, not a law']], ex: 'Multiply the batch size and you get fewer, cleaner updates.' },
      ],
      cards: [
        { id: 'opt-noise-batch', front: 'What happens to gradient noise as batch size increases?', back: 'It decreases.' },
        { id: 'opt-batch-hook', front: 'Bigger batch → ?', back: 'Fewer, cleaner updates.' },
        { id: 'opt-updates', front: '32,000 examples, batch size 32. Updates per epoch?', back: '<b>1,000</b>.', gen: gen.cUpdates },
        { id: 'opt-lr-batch', front: 'Larger learning rate with larger batch: rule or heuristic?', back: 'A useful heuristic, not a universal law.' },
      ],
    },
    {
      id: 'opt-5',
      title: 'Why SGD works on neural nets',
      steps: [
        { t: 'learn', title: 'No guarantees, still works', body: '<p>Convex guarantees are gone, but SGD still works well:</p><ul><li><b>Local directions are enough.</b> Mini-batches give useful descent directions; we don’t need a map of the whole landscape.</li><li><b>Noise helps.</b> It pushes the parameters away from saddle points and out of flat regions.</li><li><b>Many good solutions.</b> Large networks have lots of low-loss minima; we don’t need a unique global one.</li><li><b>The goal is generalization,</b> not the lowest training loss.</li></ul>' },
        { t: 'learn', title: 'Saddle points', body: '<p>At a saddle, ∇L = 0, but the loss goes up in some directions and down in others:</p><div class="eq">L(x, y) = x² − y² at (0, 0)</div><p>In a network with millions of directions, saddle-like stationary points are common. SGD noise nudges the parameters off them and exposes a way down.</p><div class="eq hook">Saddle = zero slope, but downhill still exists.</div>' },
        { t: 'num', q: 'L(x, y) = x² − y² has ∇L = 0 at (0, 0), where L = 0. What is L at (0, 1)?', a: -1, tol: 0.001, ex: '0² − 1² = −1: lower than at (0, 0). Along y the loss goes downhill, even though the slope at the saddle is zero.', gen: gen.saddleValue },
        { t: 'num', q: 'L(x, y) = x² − y². What is ∂L/∂y at (1, 2)?', a: -4, tol: 0, ex: '∂L/∂y = −2y = −2 × 2 = −4.', gen: gen.saddleGrad },
        { t: 'mc', q: 'How can SGD noise help near a saddle point?', o: ['It nudges the parameters off the saddle, exposing a downhill direction', 'It makes the gradient exactly zero', 'It turns the saddle into a minimum', 'It stops training early'], ex: 'Exactly at the saddle the gradient is zero, so plain gradient descent stalls. A small random push moves the parameters slightly off it, where the gradient is no longer zero and points the way down.' },
        { t: 'mc', q: 'Model A has the lowest training loss. Model B has slightly higher training loss but better validation performance. Which do you prefer?', o: ['B: training loss is a proxy; we care about generalization', 'A: lowest training loss wins', 'Neither', 'Whichever trained longer'], ex: 'The point of training is performance on unseen data. A slightly worse fit that generalizes better is the better model.' },
        { t: 'mc', q: 'Why don’t we need SGD to find the one global minimum of a large network?', o: ['Large networks have many good low-loss solutions', 'Neural-network losses are convex', 'SGD always finds it anyway', 'Global minima don’t exist'], ex: 'Over-parameterized networks have many solutions that fit well; landing in any of them, especially one that generalizes, is the goal.' },
        { t: 'explain', kind: 'defend', secs: 45, q: 'Why can SGD work on non-convex neural networks?', model: '<p>Neural-network losses are non-convex, so SGD has no guarantee of finding a global optimum. But mini-batches give cheap, noisy, unbiased estimates of the full gradient, which allows frequent updates. That stochasticity also helps move through saddle points and flat regions. Over-parameterized networks have many good low-loss solutions, and the real objective is a model that generalizes, not one with the absolute minimum training loss.</p>', rubric: ['Non-convex: no guarantee of a global optimum', 'Mini-batch gradients are cheap, noisy, unbiased estimates, so updates are frequent', 'Noise helps escape saddle points and flat regions', 'Many good low-loss solutions exist', 'The goal is generalization, not minimum training loss'] },
      ],
      cards: [
        { id: 'opt-noise-helps', front: 'Why can SGD noise help optimization?', back: 'It perturbs the parameters away from saddle points and flat regions.' },
        { id: 'opt-saddle', front: 'Saddle point in one line?', back: 'Zero slope, but downhill still exists.' },
        { id: 'opt-train-loss', front: 'Why not just prefer the model with the lowest training loss?', back: 'Training loss is a proxy; we care about generalization to unseen data.' },
        { id: 'opt-sgd-interview', kind: 'defend', secs: 45, front: 'Why can SGD work on non-convex neural networks?', back: 'No convex guarantee, but cheap, unbiased, noisy gradients allow frequent updates; noise escapes saddles and flat regions; many good solutions exist; the goal is generalization.', rubric: ['No global-optimum guarantee', 'Cheap, unbiased, noisy gradients', 'Noise escapes saddles and flat regions', 'Generalization is the goal'] },
      ],
    },
    {
      id: 'opt-6',
      title: 'Bias vs variance review',
      steps: [
        { t: 'learn', title: 'Three patterns', body: '<p>Assuming both errors are measured the same way and the validation set is representative:</p><ul><li><b>Low train + low-ish val:</b> healthy. Some gap is expected.</li><li><b>Low train + high val:</b> high variance (overfitting). Make memorization harder: weight decay, dropout, early stopping, a smaller model, more data, augmentation.</li><li><b>High train + high val:</b> high bias (underfitting). Make fitting easier: more capacity, train longer if not converged, less regularization, better architecture or features.</li></ul><div class="eq hook">Bias: make fitting easier. Variance: make memorization harder.</div>' },
        { t: 'mc', q: 'Training error 2%, validation error 18%, both measured the same way. What does this most likely suggest?', o: ['High variance (overfitting)', 'High bias (underfitting)', 'Healthy: a small, expected gap', 'Nothing: you can’t read anything from errors'], ex: 'Low training error, much higher validation error: the model fits its training data but doesn’t generalize. Make memorization harder.', gen: gen.diagnose },
        { t: 'mc', q: 'Training error 25%, validation error 27%, both measured the same way. What does this most likely suggest?', o: ['High bias (underfitting)', 'High variance (overfitting)', 'Healthy: a small, expected gap', 'Nothing: you can’t read anything from errors'], ex: 'Both high and close: the model can’t even fit its training set. Make fitting easier.', gen: gen.diagnose },
        { t: 'mc', q: 'Training error 3%, validation error 5%, both measured the same way. What does this most likely suggest?', o: ['Healthy: a small, expected gap', 'High variance (overfitting)', 'High bias (underfitting)', 'Nothing: you can’t read anything from errors'], ex: 'Some gap is expected. Both errors are low and the gap is small: healthy.', gen: gen.diagnose },
        { t: 'match', q: 'Match each pattern or problem to its response.', p: [['Low train, high val', 'High variance'], ['High train, high val', 'High bias'], ['Variance', 'Make memorization harder'], ['Bias', 'Make fitting easier']], ex: 'Read the training error first: high means bias. Then the gap: large means variance.' },
        { t: 'mc', q: '“Add dropout.” Which problem does that address?', o: ['High variance (overfitting)', 'High bias (underfitting)', 'Neither: it only changes training speed'], ex: 'It makes memorization harder: the variance side.', gen: gen.fixFor },
        { t: 'mc', q: '“Make the model bigger.” Which problem does that address?', o: ['High bias (underfitting)', 'High variance (overfitting)', 'Neither: it only changes training speed'], ex: 'It makes fitting easier: the bias side.', gen: gen.fixFor },
        { t: 'mc', q: '“Reduce weight decay.” Which problem does that address?', o: ['High bias (underfitting)', 'High variance (overfitting)', 'Neither: it only changes training speed'], ex: 'It makes fitting easier: the bias side.', gen: gen.fixFor },
        { t: 'explain', kind: 'diagnose', secs: 60, q: 'Training error 2%, validation error 15%. What is likely happening, and what would you try first?', model: '<p>Most likely <b>high variance</b>: the model fits its training data but doesn’t generalize. First make sure the gap is real: both errors computed the same way (eval mode, same preprocessing), and a validation set that is representative and big enough. Then make memorization harder: weight decay, dropout, data augmentation or more data, early stopping, or a smaller model, watching validation error as you go.</p>', rubric: ['Names high variance / overfitting', 'Checks the gap is real: same measurement, representative validation', 'Proposes at least two variance fixes', 'Judges by validation performance, not training loss'] },
      ],
      cards: [
        { id: 'opt-diag-var', front: 'Low training error, high validation error?', back: 'High variance (overfitting).' },
        { id: 'opt-diag-bias', front: 'High training error, high validation error?', back: 'High bias (underfitting).' },
        { id: 'opt-fix-var', front: 'Variance → ?', back: 'Make memorization harder: weight decay, dropout, early stopping, smaller model, more data, augmentation.' },
        { id: 'opt-fix-bias', front: 'Bias → ?', back: 'Make fitting easier: more capacity, train longer, less regularization, better features or architecture.' },
        { id: 'opt-healthy', front: 'What does a healthy train/validation pattern look like?', back: 'Low training error, slightly higher but still good validation error.' },
        { id: 'opt-diag-drill', front: 'Training error 2%, validation error 18%. Diagnosis?', back: '<b>High variance (overfitting)</b>. Make memorization harder.', gen: gen.cDiagnose },
      ],
    },
  ],
};
