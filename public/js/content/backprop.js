// Unit: autodiff engines and optimizers. Intuition and engine design, not long derivative chains.
// Backprop is walked one node at a time on small pictured graphs. Each picture carries every value
// and every gradient found so far, so no step asks you to hold the whole graph in your head.
// Order: concepts, then pictured walks, then engine design, then optimizers.
import { num, term, list, big, py } from '../vary.js';

// Fresh numbers every time a question is shown (see ../vary.js). Steps marked walk share one seed
// on a lesson's first pass, so a walk keeps the same graph from node to node. py is checked by
// scripts/verify_numpy.py, with deriv() its complex-step derivative.

// ---------- pictured computation graphs ----------
const NW = 72, NH = 52;
// nodes: { id, x, y, t: title, v: value, g: gradient (number, string, or undefined for "?"), s: 'cur' | 'done' }
function graph(nodes, edges, h) {
  const by = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const shown = (g) => (g === undefined ? '?' : typeof g === 'string' ? g : num(g));
  const line = ([a, b]) => `<line class="e" x1="${by[a].x + NW}" y1="${by[a].y + NH / 2}" x2="${by[b].x - 3}" y2="${by[b].y + NH / 2}" marker-end="url(#cgA)"/>`;
  const box = (n) => `<g class="nd ${n.s || ''}"><rect x="${n.x}" y="${n.y}" width="${NW}" height="${NH}" rx="10"/>`
    + `<text class="t" x="${n.x + NW / 2}" y="${n.y + 16}">${n.t}</text>`
    + `<text x="${n.x + NW / 2}" y="${n.y + 31}">${n.v === undefined ? '' : `val ${num(n.v)}`}</text>`
    + `<text class="g" x="${n.x + NW / 2}" y="${n.y + 45}">grad ${shown(n.g)}</text></g>`;
  const said = nodes.map((n) => `${n.t}${n.v === undefined ? '' : `, value ${num(n.v)}`}, gradient ${n.g === undefined ? 'unknown' : shown(n.g)}${n.s === 'cur' ? ' (this step)' : ''}`).join('; ');
  return `<svg class="cg" viewBox="0 0 340 ${h}" role="img" aria-label="Computation graph. ${said}."><defs><marker id="cgA" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path class="ah" d="M0,0L8,4L0,8z"/></marker></defs>${edges.map(line).join('')}${nodes.map(box).join('')}</svg>`;
}
// Marks the walk's progress: nodes before `at` are done (gradient shown), `at` is this step.
const progress = (order, at, grads, nodes) => nodes.map((n) => {
  const i = order.indexOf(n.id), k = order.indexOf(at);
  return i >= 0 && i < k ? { ...n, g: grads[n.id], s: 'done' } : n.id === at ? { ...n, s: 'cur' } : n;
});

// A neuron: p = x·w, z = p + b, L = z². Backward order: L, z, b, p, w, x.
const N_ORDER = ['L', 'z', 'b', 'p', 'w', 'x'];
function neuron(S) {
  const p = S.x * S.w, z = p + S.b, gz = 2 * z;
  const vals = { x: S.x, w: S.w, b: S.b, p, z, L: z * z };
  const grads = { L: 1, z: gz, b: gz, p: gz, w: gz * S.x, x: gz * S.w };
  const nodes = [
    { id: 'x', x: 0, y: 4, t: 'x' }, { id: 'w', x: 0, y: 64, t: 'w' }, { id: 'b', x: 0, y: 124, t: 'b' },
    { id: 'p', x: 89, y: 34, t: 'p = x·w' }, { id: 'z', x: 178, y: 74, t: 'z = p + b' }, { id: 'L', x: 267, y: 74, t: 'L = z²' },
  ].map((n) => ({ ...n, v: vals[n.id] }));
  const edges = [['x', 'p'], ['w', 'p'], ['p', 'z'], ['b', 'z'], ['z', 'L']];
  return { ...S, ...vals, grads, fig: (at) => graph(at ? progress(N_ORDER, at, grads, nodes) : nodes, edges, 180) };
}
function pickNeuron(v) {
  for (;;) {
    const S = { x: v.pick([-2, -1, 1, 2, 3]), w: v.nz(-3, 3), b: v.pick([-2, -1, 1, 2]) };
    if (S.x * S.w + S.b !== 0) return S;
  }
}
const fullN = (N) => `lambda x, w, b: (x * w + b)**2, ${N.x}, ${N.w}, ${N.b}`;
function neuronStep(at, N) {
  const g = N.grads;
  const s = {
    L: { q: 'Backward starts at the loss. What is L’s gradient, ∂L/∂L?', a: 1,
      ex: 'Nudge L by a tiny ε and L changes by exactly ε, so ∂L/∂L = 1. Every other gradient is built from this 1.', py: `ans = deriv(lambda L: L, ${N.L})` },
    z: { q: 'L = z², so its local derivative is 2z. L’s gradient is 1. What is z’s gradient?', a: g.z,
      ex: `Incoming × local: 1 × 2z = 1 × 2 × ${term(N.z)} = ${num(g.z)}. The rule needs z’s value, cached from the forward pass.`, py: `ans = deriv(lambda z: z**2, ${N.z})` },
    b: { q: 'z = p + b. A sum’s local derivative is 1 for each input. What is b’s gradient?', a: g.b,
      ex: `Incoming × local: ${num(g.z)} × 1 = ${num(g.b)}. A sum hands its gradient to each input unchanged.`, py: `ans = deriv(${fullN(N)}, wrt=2)` },
    p: { q: 'Still at the sum z = p + b. What is p’s gradient?', a: g.p,
      ex: `Same rule as for b: ${num(g.z)} × 1 = ${num(g.p)}. Both inputs of a sum get exactly the sum’s gradient.`, py: `ans = deriv(lambda p: (p + ${N.b})**2, ${N.p})` },
    w: { q: 'p = x·w. Its local derivative for w is x’s value. What is w’s gradient?', a: g.w,
      ex: `Incoming × local: ${num(g.p)} × x = ${num(g.p)} × ${term(N.x)} = ${num(g.w)}. The product rule reads x’s cached forward value, not w’s.`, py: `ans = deriv(${fullN(N)}, wrt=1)` },
    x: { q: 'Last input of p = x·w. Its local derivative for x is w’s value. What is x’s gradient?', a: g.x,
      ex: `Incoming × local: ${num(g.p)} × w = ${num(g.p)} × ${term(N.w)} = ${num(g.x)}. Each input of a product gets the other input’s value.`, py: `ans = deriv(${fullN(N)}, wrt=0)` },
  }[at];
  return { ...s, fig: N.fig(at) };
}

// Two paths: a = c·x, b = x², y = a + b, L = k·y. x's gradient accumulates from both.
const M_ORDER = ['L', 'y', 'a', 'b', 'x'];
function merge(S) {
  const a = S.c * S.x, b = S.x * S.x, y = a + b;
  const vals = { x: S.x, a, b, y, L: S.k * y };
  const fromA = S.k * S.c, fromB = S.k * 2 * S.x;
  const grads = { L: 1, y: S.k, a: S.k, b: S.k, x: fromA + fromB };
  const nodes = [
    { id: 'x', x: 0, y: 64, t: 'x' }, { id: 'a', x: 89, y: 14, t: `a = ${S.c}x` }, { id: 'b', x: 89, y: 114, t: 'b = x²' },
    { id: 'y', x: 178, y: 64, t: 'y = a + b' }, { id: 'L', x: 267, y: 64, t: `L = ${S.k}y` },
  ].map((n) => ({ ...n, v: vals[n.id] }));
  const edges = [['x', 'a'], ['x', 'b'], ['a', 'y'], ['b', 'y'], ['y', 'L']];
  const fig = (at, xg) => graph(at ? progress(M_ORDER, at, grads, nodes).map((n) => (n.id === 'x' && xg !== undefined ? { ...n, g: xg } : n)) : nodes, edges, 170);
  return { ...S, ...vals, grads, fromA, fromB, fig };
}
const pickMerge = (v) => ({ x: v.nz(-3, 3), c: v.pick([2, 3, 4]), k: v.pick([2, 3, 4]) });
function mergeStep(at, M) {
  const s = {
    y: { q: `L = ${M.k}y, so its local derivative is ${M.k}. L’s gradient is 1. What is y’s gradient?`, a: M.k,
      ex: `Incoming × local: 1 × ${M.k} = ${M.k}.`, py: `ans = deriv(lambda y: ${M.k} * y, ${M.y})`, fig: M.fig('y') },
    a: { q: 'y = a + b. What is a’s gradient?', a: M.k,
      ex: `A sum passes its gradient through: ${M.k} × 1 = ${M.k}.`, py: `ans = deriv(lambda a: ${M.k} * (a + ${M.b}), ${M.a})`, fig: M.fig('a') },
    x1: { q: `Now x. Nothing has arrived yet, so x.grad starts at 0. First a = ${M.c}x sends incoming × local. What does x.grad hold after a’s contribution?`, a: M.fromA,
      ex: `a’s gradient is ${M.k} and its local derivative for x is ${M.c}: ${M.k} × ${M.c} = ${num(M.fromA)}. x.grad = 0 + ${num(M.fromA)}.`,
      py: `x0 = ${M.x}\nans = deriv(lambda x: ${M.k} * (${M.c} * x + x0**2), ${M.x})`, fig: M.fig('x', '0+?') },
    x2: { q: `x.grad holds ${num(M.fromA)} from the path through a. Now b = x² sends its contribution, and the engine does x.grad += …. What does x.grad hold now?`, a: M.fromA + M.fromB,
      ex: `b sends ${M.k} × 2x = ${M.k} × ${term(2 * M.x)} = ${num(M.fromB)}. Accumulate: ${num(M.fromA)} + ${term(M.fromB)} = ${num(M.fromA + M.fromB)}. Two paths, so two contributions add.`,
      py: `ans = deriv(lambda x: ${M.k} * (${M.c} * x + x**2), ${M.x})`, fig: M.fig('x', `${num(M.fromA)}+?`) },
    bug: { q: `Same graph, but the engine overwrites (x.grad = …) instead of adding. a’s contribution arrives first, then b’s. What does x.grad end up as?`, a: M.fromB,
      ex: `The second write replaces the first, leaving b’s ${num(M.fromB)}. The true gradient is ${num(M.fromA + M.fromB)}: the path through a was silently lost.`,
      py: `g = 0.0\ng = ${M.fromA}\ng = ${M.fromB}\nans = g`, fig: M.fig('x', '?') },
  }[at];
  return s;
}

// Big parameter counts in words: 7 billion, 125 million.
const words = (n) => (n >= 1e9 ? `${num(n / 1e9)} billion` : `${num(n / 1e6)} million`);

const gen = {
  passes(v) {
    const n = v.pick([1e6, 125e6, 7e9, 50000]), rev = v.int(0, 1) === 1;
    return { q: `A model has ${n >= 1e6 ? words(n) : big(n)} parameters and one scalar loss. How many ${rev ? 'reverse' : 'forward'}-mode passes give the full gradient?`, a: rev ? 1 : n,
      ex: rev ? `One. A single backward pass from ∂L/∂L = 1 reaches every parameter. Forward mode would need one pass per parameter.`
        : `One per parameter: forward mode carries the derivative of a single input per pass. Reverse mode needs just one pass, because there is one output.`,
      py: `ans = ${rev ? 1 : n}` };
  },
  neuron: (at) => (v) => neuronStep(at, neuron(pickNeuron(v))),
  merge: (at) => (v) => mergeStep(at, merge(pickMerge(v))),
  stale(v) {
    const g1 = v.nz(-5, 8), g2 = v.nz(-5, 8);
    return { q: `Step 1’s backward left w.grad = ${num(g1)}. Before step 2, zero_grad() is skipped. Step 2’s backward computes ${num(g2)} for w. What does w.grad hold now?`, a: g1 + g2,
      ex: `Backward adds into .grad, so step 1’s ${num(g1)} is still there: ${num(g1)} + ${term(g2)} = ${num(g1 + g2)}. The optimizer would step on a stale mix. Reset grads between optimizer steps.`,
      py: `g = 0.0\ng += ${g1}\ng += ${g2}\nans = g` };
  },
  effBatch(v) {
    const m = v.pick([4, 8, 16, 32, 64]), k = v.pick([2, 4, 8, 16]);
    return { q: `Micro-batches of ${m}, with ${k} backward passes accumulated before each optimizer step. What batch size does that simulate?`, a: m * k,
      ex: `${m} × ${k} = ${m * k}. Gradients add up across the micro-batches, and one step uses all of them, like a batch of ${m * k} that doesn’t have to fit in memory at once.`,
      py: `ans = ${m} * ${k}` };
  },
  accumMean(v) {
    const gs = v.vec(4, -6, 9), mean = gs.reduce((s, x) => s + x, 0) / 4, sum = gs.reduce((s, x) => s + x, 0);
    return { q: `Four micro-batches give mean-loss gradients ${list(gs)} for one weight. Each micro-batch’s loss is divided by 4 before backward, and grads aren’t zeroed in between. What is the accumulated gradient?`, a: mean,
      ex: `Each backward adds g/4, so the total is (${gs.map(term).join(' + ')}) / 4 = ${num(mean)}: the mean over the whole big batch. Without the /4 you’d get the sum, ${num(sum)}: four times too big a step.`,
      py: `g = 0.0\nfor x in ${py(gs)}:\n    g += x / 4\nans = g` };
  },
  sgd(v) {
    const w = v.pick([-2, -1, 0.5, 1, 2]), g = v.nz(-6, 6), eta = v.pick([0.1, 0.01, 0.5]), w1 = +(w - eta * g).toFixed(6);
    return { q: `Plain SGD, η = ${num(eta)}. w = ${num(w)} and its gradient is ${num(g)}. What is w after the step?`, a: w1,
      ex: `w − η·g = ${num(w)} − ${num(eta)} × ${term(g)} = ${num(w1)}. SGD has no memory: the step is just the gradient, scaled.`, py: `ans = ${w} - ${eta} * ${g}` };
  },
  velocity(v) {
    const vp = v.pick([0, 1, 2, -1, 0.5]), g = v.nz(-4, 4), nv = +(0.9 * vp + g).toFixed(6);
    return { q: `SGD with momentum 0.9: v ← 0.9·v + g, then w ← w − η·v. The buffer holds v = ${num(vp)} and the new gradient is ${num(g)}. What is the new v?`, a: nv,
      ex: `0.9 × ${term(vp)} + ${term(g)} = ${num(nv)}. v is a decaying running sum of gradients: steady gradients build speed, alternating ones cancel out.`,
      py: `v = ${vp}\nv = 0.9 * v + ${g}\nans = v` };
  },
  momentumStep(v) {
    const w = v.pick([1, 2, -1]), vp = v.pick([1, 2, -1, 0.5]), g = v.nz(-4, 4), eta = v.pick([0.1, 0.5]), nv = 0.9 * vp + g, w1 = +(w - eta * nv).toFixed(6);
    return { q: `Momentum 0.9, η = ${num(eta)}. w = ${num(w)}, the buffer holds v = ${num(vp)}, and the new gradient is ${num(g)}. What is w after the step?`, a: w1,
      ex: `First v = 0.9 × ${term(vp)} + ${term(g)} = ${num(nv)}. Then w − η·v = ${num(w)} − ${num(eta)} × ${term(nv)} = ${num(w1)}. Plain SGD would have used only the gradient, ${num(g)}: same gradient, different update.`,
      py: `v = 0.9 * ${vp} + ${g}\nans = ${w} - ${eta} * v` };
  },
  lrStep(v) {
    const eta = v.pick([0.001, 0.1, 0.25, 0.75, 1.25]), w1 = +(1 - 2 * eta).toFixed(6);
    return { q: `L(w) = w², so the gradient is 2w. From w = 1 with η = ${num(eta)}, what is w after one step?`, a: w1,
      ex: `w − η·2w = 1 − 2 × ${num(eta)} = ${num(w1)}. ${Math.abs(w1) > 1 ? 'It overshot the minimum at 0 by more than it started: the steps will grow.' : w1 < 0 ? 'It jumped past the minimum at 0 to the other side.' : w1 > 0.99 ? 'Barely moved: at this rate it takes thousands of steps.' : 'It moved a sensible fraction of the way toward the minimum at 0.'}`,
      py: `ans = 1 - ${eta} * deriv(lambda w: w**2, 1.0)` };
  },
  lrRegime(v) {
    const [eta, k] = v.pick([[0.1, 'ok'], [0.25, 'ok'], [0.4, 'ok'], [0.7, 'osc'], [0.9, 'osc'], [1.1, 'div'], [1.5, 'div'], [0.001, 'slow'], [0.002, 'slow']]);
    const label = { ok: 'It shrinks steadily toward the minimum', osc: 'It flips sign each step but still shrinks', div: 'It grows every step: training diverges', slow: 'It barely moves: progress is very slow' };
    const f = 1 - 2 * eta;
    return { q: `L(w) = w², gradient descent from w = 1 with η = ${num(eta)}. Each step multiplies w by 1 − 2η = ${num(f)}. What happens over many steps?`,
      o: [label[k], ...Object.keys(label).filter((x) => x !== k).map((x) => label[x])],
      ex: k === 'div' ? `|${num(f)}| > 1, so every step overshoots by more than the last: too large a learning rate diverges.` : k === 'osc' ? `${num(f)} is negative but bigger than −1: it overshoots to the other side each step, yet the size still shrinks.` : k === 'slow' ? `${num(f)} is just under 1: each step removes only a sliver. Too small a learning rate wastes steps.` : `${num(f)} is between 0 and 1: each step closes a fixed fraction of the gap.` };
  },
  adamM(v) {
    const mp = v.pick([0, 0.5, 1, -1, 2]), g = v.nz(-4, 4), m = +(0.9 * mp + 0.1 * g).toFixed(6);
    return { q: `Adam’s first moment, β₁ = 0.9: m ← 0.9·m + 0.1·g. With m = ${num(mp)} and a new gradient of ${num(g)}, what is m?`, a: m,
      ex: `0.9 × ${term(mp)} + 0.1 × ${term(g)} = ${num(m)}. m is a moving average of gradients: it tracks the direction they keep pointing.`,
      py: `m = ${mp}\nm = 0.9 * m + 0.1 * ${g}\nans = m` };
  },
  adamV(v) {
    const vp = v.pick([0, 0.5, 1, 4]), g = v.pick([1, 2, 3, -2, -1, 0.5]), nv = +(0.999 * vp + 0.001 * g * g).toFixed(8);
    return { q: `Adam’s second moment, β₂ = 0.999: v ← 0.999·v + 0.001·g². With v = ${num(vp)} and a new gradient of ${num(g)}, what is v?`, a: nv, tol: 0.00001,
      ex: `0.999 × ${num(vp)} + 0.001 × ${term(g)}² = ${num(nv, 8)}. v averages squared gradients: it tracks their size, and the sign of g makes no difference.`,
      py: `v = ${vp}\nv = 0.999 * v + 0.001 * (${g})**2\nans = v` };
  },
  adamFirst(v) {
    const g = v.pick([-3, -0.2, -0.01, 0.004, 0.5, 4, 25]), eta = v.pick([0.1, 0.01, 0.001]), d = -eta * Math.sign(g);
    return { q: `Adam’s very first step, η = ${num(eta)}, for a weight whose gradient is ${num(g)}. By how much does the weight change? (Ignore ε.)`, a: d, tol: 0.000001,
      ex: `On step 1, m̂ = g and v̂ = g², so the change is −η·g/|g| = ${num(d)}, whatever the size of g. SGD with the same η would change it by −η·g = ${num(-eta * g)}.`,
      py: `g, eta, b1, b2 = ${g}, ${eta}, 0.9, 0.999\nm, s = (1 - b1) * g, (1 - b2) * g * g\nans = -eta * (m / (1 - b1)) / np.sqrt(s / (1 - b2))` };
  },
  adamMem(v) {
    const n = v.pick([125e6, 1.3e9, 7e9, 13e9, 70e9]), gb = (n * 8) / 1e9;
    return { q: `A model has ${words(n)} parameters. Adam keeps m and v for each one in fp32 (4 bytes each). How many GB is that optimizer state?`, a: +gb.toFixed(3), tol: +(0.02 * gb).toFixed(3),
      ex: `${words(n)} × 2 values × 4 bytes = ${num(gb)} GB, twice the size of the fp32 weights themselves.`, py: `ans = ${n} * 2 * 4 / 1e9` };
  },
  adamTotal(v) {
    const n = v.pick([125e6, 1.3e9, 7e9, 13e9]), gb = (n * 16) / 1e9;
    return { q: `${words(n)} parameters, all fp32. Counting the weights, their gradients, and Adam’s m and v, how many GB before any activations?`, a: +gb.toFixed(3), tol: +(0.02 * gb).toFixed(3),
      ex: `4 copies × 4 bytes = 16 bytes per parameter: ${words(n)} × 16 = ${num(gb)} GB. Adam’s state is half of it.`, py: `ans = ${n} * (4 + 4 + 4 + 4) / 1e9` };
  },
};

// Written examples for the walk steps: the same graphs with fixed numbers.
const N0 = neuron({ x: 2, w: 3, b: 1 }), M0 = merge({ x: 2, c: 3, k: 2 });
const walkN = (at) => ({ t: 'num', walk: true, tol: 0, ...neuronStep(at, N0), gen: gen.neuron(at) });
const walkM = (at) => ({ t: 'num', walk: true, tol: 0, ...mergeStep(at, M0), gen: gen.merge(at) });
const CHAIN = graph([
  { id: 'x', x: 0, y: 6, t: 'x', v: 2 }, { id: 'a', x: 134, y: 6, t: 'a = 3x', v: 6 }, { id: 'L', x: 268, y: 6, t: 'L = a²', v: 36, g: 1, s: 'cur' },
], [['x', 'a'], ['a', 'L']], 64);

export default {
  id: 'bp',
  title: 'Autodiff & Adam',
  sub: 'How an engine computes gradients, and how optimizers turn them into steps.',
  c: 'bp',
  guide: `<h3>Memory hooks</h3><div class="eq hook">At each edge: incoming gradient × local derivative.</div><div class="eq hook">Along a path: multiply. Where paths merge: add.</div><div class="eq hook">Forward = values. Backward = sensitivities.</div><div class="eq hook">Autodiff computes gradients. The optimizer decides the step.</div><div class="eq hook">Adam: first moment = direction, second moment = scale.</div>
  <h3>Reverse mode</h3><ul><li>One scalar loss, many parameters: one backward pass gives every gradient. Forward mode needs one pass per input.</li><li>Start at the loss with ∂L/∂L = 1: nudging L by ε changes L by ε.</li><li>Visit nodes in reverse topological order, so each node’s gradient is complete before it passes it on.</li></ul>
  <h3>The engine</h3><ul><li>Each node stores its value, its grad (starting at 0), its parents, and a local backward rule.</li><li>Backward rules read cached forward values (a product needs the other input; ReLU needs the sign). That is why training holds activations in memory.</li><li>Grads accumulate with +=, because a node on several paths must add every contribution.</li><li>Reset grads between optimizer steps, or old gradients leak into the next step.</li><li>Gradient accumulation on purpose: k backward passes on micro-batches (each loss ÷ k), then one step, simulates a batch k times larger.</li></ul>
  <h3>Optimizers</h3><ul><li>SGD: w ← w − η·g. Momentum: v ← 0.9·v + g, w ← w − η·v.</li><li>Adam: m ← β₁m + (1 − β₁)g (direction), v ← β₂v + (1 − β₂)g² (scale), step ≈ η·m̂/√v̂. Per-parameter step sizes, nearly independent of the gradient’s scale.</li><li>Same gradient, different updates: each optimizer keeps different state and transforms the gradient differently.</li><li>Adam stores m and v for every parameter: twice the weights again. fp32 weights + grads + m + v = 16 bytes per parameter.</li><li>Learning rate: too large overshoots or diverges; too small crawls.</li></ul>`,
  lessons: [
    {
      id: 'bp-1',
      title: 'Why reverse mode',
      steps: [
        { t: 'learn', title: 'One loss, many parameters', body: '<p>Training has <b>one scalar loss</b> and millions of parameters. Reverse mode starts at the loss and sweeps backward once, and that single pass gives the gradient for <b>every</b> parameter.</p><p>Forward mode pushes derivatives forward from one input at a time: one pass per parameter. Great for few inputs and many outputs; hopeless for a network.</p>' },
        { t: 'mc', q: 'Why is reverse-mode autodiff the right tool for training a neural network?', o: ['One scalar loss, many parameters: one backward pass gives every gradient', 'It is more accurate than forward mode', 'It doesn’t need a forward pass', 'It avoids the chain rule'], ex: 'Both modes are exact. The difference is cost: reverse mode is one pass per output, and a loss is a single output.' },
        { t: 'num', q: 'A model has 7 billion parameters and one scalar loss. How many reverse-mode passes give the full gradient?', a: 1, tol: 0, ex: 'One. A single backward pass from ∂L/∂L = 1 reaches every parameter. Forward mode would need one pass per parameter.', gen: gen.passes },
        { t: 'learn', title: 'Why start at 1', body: `${CHAIN}<p>The backward pass asks, for every node, “if this changed a little, how much would L change?” For L itself the answer is exact: nudge L by ε and L changes by ε. So <b>∂L/∂L = 1</b>, and every other gradient is that 1 multiplied by local derivatives on the way back.</p>` },
        { t: 'mc', q: 'Why does the loss’s own gradient start at 1?', o: ['Nudging L by ε changes L by exactly ε', 'It’s a convention that could be any number', 'Because the learning rate is 1', 'Because L is usually near 1'], ex: '∂L/∂L is a real derivative, and it is exactly 1. If you seeded it with 2, every gradient in the graph would come out doubled.' },
        { t: 'match', q: 'Match each backprop rule to where it applies.', p: [['At each edge', 'Incoming gradient × local derivative'], ['Along a path', 'Multiply'], ['Where paths merge', 'Add'], ['At the loss', 'Start with 1']], ex: 'Three rules and one starting point carry the whole backward pass.' },
        { t: 'tf', q: 'Reverse mode computes each parameter’s gradient in its own separate pass.', a: false, ex: 'That is forward mode’s cost model. Reverse mode shares one backward sweep across all parameters.' },
      ],
      cards: [
        { id: 'bp-why-reverse', front: 'Why is reverse mode efficient for training?', back: 'One backward pass gives every parameter’s gradient.' },
        { id: 'bp-seed', front: 'Why does the loss’s gradient start at one?', back: 'Nudging the loss changes it by exactly that nudge.' },
        { id: 'bp-rules', front: 'What happens at each edge in backprop?', back: 'Incoming gradient times local derivative.' },
        { id: 'bp-forward-mode', front: 'When does forward mode win?', back: 'Few inputs, many outputs.' },
      ],
    },
    {
      id: 'bp-2',
      title: 'One node at a time',
      steps: [
        { t: 'learn', title: 'A neuron, forward pass done', body: `${N0.fig()}<p>The forward pass computed every value and recorded the graph. Now go backward from L, <b>one node at a time</b>. At each edge: <b>incoming gradient × local derivative</b>. The picture keeps track of everything found so far.</p>` },
        walkN('L'), walkN('z'), walkN('b'), walkN('p'), walkN('w'), walkN('x'),
        { t: 'mc', q: 'Why did the step for w need x’s value?', o: ['The product p = x·w has local derivative x for w, so the backward rule reads x’s cached forward value', 'Because x is w’s parent', 'Because x was processed first', 'It didn’t: w’s gradient never depends on x'], ex: 'Backward rules use forward values. That is why an engine caches activations during the forward pass, and why training needs more memory than inference.' },
        { t: 'mc', q: 'b and p both received exactly z’s gradient. Why?', o: ['z = p + b is a sum: its local derivatives are 1, so it passes its gradient through unchanged', 'Because b and p have the same value', 'Because they were processed together', 'Coincidence'], ex: 'A sum node copies its incoming gradient to every input. A product node scales it by the other input instead.' },
      ],
      cards: [
        { id: 'bp-walk-order', front: 'Where does the backward pass start?', back: 'At the loss.' },
        { id: 'bp-sum-node', front: 'What does a sum node send back?', back: 'Its incoming gradient, unchanged.' },
        { id: 'bp-mul-node', front: 'What scales the gradient through a product node?', back: 'The other input’s value.' },
        { id: 'bp-cache', front: 'Why does backward need forward values?', back: 'Local derivatives depend on them.' },
      ],
    },
    {
      id: 'bp-3',
      title: 'When paths merge',
      steps: [
        { t: 'learn', title: 'x feeds two nodes', body: `${M0.fig()}<p>x reaches L along two paths, through a and through b. Each path sends x a contribution, and the engine <b>adds</b> them into x.grad with <code>+=</code>.</p><div class="eq hook">Along a path: multiply. Where paths merge: add.</div>` },
        walkM('y'), walkM('a'),
        walkM('x1'), walkM('x2'), walkM('bug'),
        { t: 'mc', q: 'If x had parents of its own, when could it run its backward rule?', o: ['Only after both a and b have added into x.grad', 'As soon as a has sent its contribution', 'Before a and b, since x comes first', 'Any time: order doesn’t matter'], ex: 'Reverse topological order guarantees every child has added its contribution before a node passes its gradient on. Going early would send an incomplete gradient upstream.' },
        { t: 'mc', q: 'Why do engines accumulate with += instead of overwriting?', o: ['A node on several paths must add every path’s contribution', 'It is faster', 'Python requires it', 'To keep gradients positive'], ex: 'Each path carries part of the total change. Overwriting keeps only the last path, which is exactly the bug from the previous step.' },
      ],
      cards: [
        { id: 'bp-accumulate', front: 'Why += for gradients?', back: 'Several paths must all add their part.' },
        { id: 'bp-merge', front: 'Two paths reach x. What happens to their gradients?', back: 'They add.' },
        { id: 'bp-topo', front: 'Why reverse topological order?', back: 'A node’s gradient must be complete before it passes it on.' },
      ],
    },
    {
      id: 'bp-4',
      title: 'Inside the engine',
      steps: [
        { t: 'learn', title: 'Four things per node', body: `<p>Every node stores its <b>value</b>, its <b>grad</b> (∂L/∂node, starting at 0), its <b>parents</b>, and a <b>local backward rule</b>.</p><pre class="py">def relu(self):
    out = Value(max(0, self.data), (self,))
    def _backward():
        # reads the cached forward value of self
        self.grad += (self.data > 0) * out.grad
    out._backward = _backward
    return out</pre><p>The rule needs <code>self.data</code> from the forward pass, so the engine keeps it. That cache is why training holds activations in memory.</p>` },
        { t: 'match', q: 'Match each part of a node to what it holds.', p: [['value', 'Result of the forward pass'], ['grad', '∂L/∂node, starts at 0'], ['parents', 'Nodes it was computed from'], ['backward rule', 'Sends incoming × local to each parent']], ex: 'Value and parents come from the forward pass; grad and the rule do the backward pass.' },
        { t: 'mc', q: 'Why does training a network use much more memory than running it?', o: ['The backward pass needs cached forward values (activations)', 'Gradients are bigger than weights', 'Training uses a larger model', 'Inference skips the weights'], ex: 'Inference can throw each activation away once it’s used. Training must keep them until backward reads them in the local rules.' },
        { t: 'learn', title: 'Reset between steps', body: '<pre class="py">loss.backward()     # autodiff: adds into every p.grad\nopt.step()          # optimizer: turns grads into updates\nopt.zero_grad()     # clear grads before the next backward</pre><p>Because backward <b>adds</b> into <code>.grad</code>, skipping the reset mixes last step’s gradient into this one.</p>' },
        { t: 'num', q: 'Step 1’s backward left w.grad = 3. Before step 2, zero_grad() is skipped. Step 2’s backward computes 5 for w. What does w.grad hold now?', a: 8, tol: 0, ex: 'Backward adds into .grad, so step 1’s 3 is still there: 3 + 5 = 8. Reset grads between optimizer steps.', gen: gen.stale },
        { t: 'learn', title: 'Accumulating on purpose', body: '<pre class="py">opt.zero_grad()\nfor micro in split(batch, 4):\n    loss = model(micro).loss / 4   # scale so the total is a mean\n    loss.backward()                # grads add up across micro-batches\nopt.step()                         # one step on the combined gradient</pre><p>The same += that forces you to zero grads also lets you <b>simulate a bigger batch</b> when it won’t fit in memory.</p>' },
        { t: 'num', q: 'Micro-batches of 8, with 4 backward passes accumulated before each optimizer step. What batch size does that simulate?', a: 32, tol: 0, ex: '8 × 4 = 32. Gradients add up across the micro-batches, and one step uses all of them.', gen: gen.effBatch },
        { t: 'num', q: 'Four micro-batches give mean-loss gradients [4, −2, 6, 0] for one weight. Each micro-batch’s loss is divided by 4 before backward, and grads aren’t zeroed in between. What is the accumulated gradient?', a: 2, tol: 0.001, ex: 'Each backward adds g/4: (4 − 2 + 6 + 0) / 4 = 2, the mean over the big batch. Without the /4 you’d get 8, four times too big a step.', gen: gen.accumMean },
        { t: 'mc', q: 'Why divide each micro-batch loss by the number of accumulation steps?', o: ['So the summed gradient equals the big batch’s mean gradient', 'To save memory', 'To make backward faster', 'It isn’t needed'], ex: 'Backward sums across micro-batches. Dividing by k turns that sum into the average you’d get from one batch k times bigger, so the learning rate means the same thing.' },
        { t: 'explain', kind: 'recall', secs: 60, q: 'What does an autodiff engine store per node, and why does the backward pass need the forward pass’s values?', model: '<p>Each node stores its <b>value</b>, its <b>grad</b> (∂L/∂node, zero until backward adds into it), its <b>parents</b>, and a <b>local backward rule</b> for its operation. The rule computes incoming gradient × local derivative for each parent, and local derivatives depend on forward values: a product needs the other input, ReLU needs the sign of its input, a square needs its input. So the forward pass caches those values, which is why training needs more memory than inference.</p>', rubric: ['Value, grad, parents and the backward rule', 'grad starts at 0 and is accumulated with +=', 'The rule is incoming gradient × local derivative', 'Local derivatives use cached forward values, with an example', 'Caching explains training’s extra memory'] },
      ],
      cards: [
        { id: 'bp-node-fields', front: 'What is a node’s grad before backward runs?', back: 'Zero.' },
        { id: 'bp-zero-grad', front: 'Why reset grads between optimizer steps?', back: 'Old gradients would leak into the next step.' },
        { id: 'bp-grad-accum', front: 'What does gradient accumulation simulate?', back: 'A bigger batch than fits in memory.' },
        { id: 'bp-train-memory', front: 'Why does training need more memory than inference?', back: 'Backward needs the cached activations.' },
      ],
    },
    {
      id: 'bp-5',
      title: 'Autodiff vs optimizer',
      steps: [
        { t: 'learn', title: 'Two separate jobs', body: '<p><b>Autodiff</b> computes the gradient: how the loss responds to each parameter. The <b>optimizer</b> decides what to do with it: how big a step, in which direction, using whatever memory it keeps.</p><div class="eq hook">Autodiff computes gradients. The optimizer decides the step.</div><p>So two optimizers can receive <b>the same gradient</b> and make <b>different updates</b>.</p>' },
        { t: 'mc', q: 'SGD and Adam get exactly the same gradient. Why can their updates differ?', o: ['Each optimizer keeps different state and transforms the gradient differently', 'They compute different gradients', 'Adam runs its own backward pass', 'They can’t: same gradient, same update'], ex: 'The gradient is the optimizer’s input, not its output. SGD scales it; momentum blends it with past gradients; Adam normalizes it per parameter.' },
        { t: 'num', q: 'Plain SGD, η = 0.1. w = 2 and its gradient is 4. What is w after the step?', a: 1.6, tol: 0.001, ex: 'w − η·g = 2 − 0.1 × 4 = 1.6. SGD has no memory: the step is just the gradient, scaled.', gen: gen.sgd },
        { t: 'learn', title: 'Momentum remembers', body: '<div class="eq">v ← 0.9·v + g     w ← w − η·v</div><p>v is a decaying running sum of past gradients. When gradients keep pointing the same way, v grows and the steps speed up. When they zig-zag, they cancel in v and the steps calm down.</p>' },
        { t: 'num', q: 'SGD with momentum 0.9: v ← 0.9·v + g, then w ← w − η·v. The buffer holds v = 2 and the new gradient is 1. What is the new v?', a: 2.8, tol: 0.001, ex: '0.9 × 2 + 1 = 2.8. Steady gradients build speed; alternating ones cancel out.', gen: gen.velocity },
        { t: 'num', q: 'Momentum 0.9, η = 0.1. w = 1, the buffer holds v = 2, and the new gradient is 1. What is w after the step?', a: 0.72, tol: 0.001, ex: 'First v = 0.9 × 2 + 1 = 2.8. Then w − η·v = 1 − 0.1 × 2.8 = 0.72. Plain SGD would have used only the gradient, 1: same gradient, different update.', gen: gen.momentumStep },
        { t: 'learn', title: 'Learning-rate intuition', body: '<p>On L(w) = w², each gradient step multiplies w by <span class="m">1 − 2η</span>.</p><ul><li>Small η: barely moves. Progress is very slow.</li><li>Moderate η: closes a steady fraction of the gap.</li><li>Large η: jumps past the minimum; too large and each jump is bigger than the last, so it diverges.</li></ul>' },
        { t: 'num', q: 'L(w) = w², so the gradient is 2w. From w = 1 with η = 1.25, what is w after one step?', a: -1.5, tol: 0.001, ex: '1 − 2 × 1.25 = −1.5. It overshot the minimum at 0 by more than it started: the steps will grow.', gen: gen.lrStep },
        { t: 'mc', q: 'L(w) = w², gradient descent from w = 1 with η = 0.001. Each step multiplies w by 1 − 2η = 0.998. What happens over many steps?', o: ['It barely moves: progress is very slow', 'It shrinks steadily toward the minimum', 'It flips sign each step but still shrinks', 'It grows every step: training diverges'], ex: '0.998 is just under 1: each step removes only a sliver. Too small a learning rate wastes steps.', gen: gen.lrRegime },
        { t: 'mc', q: 'L(w) = w², gradient descent from w = 1 with η = 1.5. Each step multiplies w by 1 − 2η = −2. What happens over many steps?', o: ['It grows every step: training diverges', 'It shrinks steadily toward the minimum', 'It flips sign each step but still shrinks', 'It barely moves: progress is very slow'], ex: '|−2| > 1, so every step overshoots by more than the last: too large a learning rate diverges.', gen: gen.lrRegime },
      ],
      cards: [
        { id: 'bp-autodiff-vs-opt', front: 'What does the optimizer do that autodiff doesn’t?', back: 'Decides the step from the gradient.' },
        { id: 'bp-same-grad', front: 'Can two optimizers step differently on the same gradient?', back: 'Yes: each keeps different state.' },
        { id: 'bp-momentum', front: 'What does momentum remember?', back: 'A decaying sum of past gradients.' },
        { id: 'bp-lr', front: 'What happens with too large a learning rate?', back: 'Steps overshoot and can diverge.' },
      ],
    },
    {
      id: 'bp-6',
      title: 'Adam',
      steps: [
        { t: 'learn', title: 'Two moving averages', body: '<p>Adam keeps two numbers for every parameter:</p><ul><li><b>m</b>, the first moment: a moving average of gradients. It says which way to go.</li><li><b>v</b>, the second moment: a moving average of <b>squared</b> gradients. It says how big the gradients usually are.</li></ul><p>The step is about η·m̂/√v̂ (after bias correction): direction divided by scale.</p><div class="eq hook">First moment = direction, second moment = scale.</div>' },
        { t: 'num', q: 'Adam’s first moment, β₁ = 0.9: m ← 0.9·m + 0.1·g. With m = 0.5 and a new gradient of 2, what is m?', a: 0.65, tol: 0.0001, ex: '0.9 × 0.5 + 0.1 × 2 = 0.65. m is a moving average of gradients: it tracks the direction they keep pointing.', gen: gen.adamM },
        { t: 'num', q: 'Adam’s second moment, β₂ = 0.999: v ← 0.999·v + 0.001·g². With v = 0.5 and a new gradient of 2, what is v?', a: 0.5035, tol: 0.00001, ex: '0.999 × 0.5 + 0.001 × 2² = 0.5035. v averages squared gradients: it tracks their size, and the sign of g makes no difference.', gen: gen.adamV },
        { t: 'mc', q: 'Parameter A always gets gradient 10; parameter B always gets 0.1. After many Adam steps, how do their step sizes compare?', o: ['About the same: m̂/√v̂ is about ±1 for both', 'A’s steps are 100× bigger', 'B’s steps are 100× bigger', 'A’s steps are 10× bigger'], ex: 'For a steady gradient g, m ≈ g and √v ≈ |g|, so the step is about η either way. Under SGD, A’s steps would be 100× B’s.' },
        { t: 'num', q: 'Adam’s very first step, η = 0.01, for a weight whose gradient is 0.004. By how much does the weight change? (Ignore ε.)', a: -0.01, tol: 0.000001, ex: 'On step 1, m̂ = g and v̂ = g², so the change is −η·g/|g| = −0.01, whatever the size of g. SGD with the same η would change it by −0.00004.', gen: gen.adamFirst },
        { t: 'learn', title: 'Adam’s memory bill', body: '<p>m and v are full-size copies, one value each per parameter. In fp32 that is 8 extra bytes per parameter, twice the weights again.</p><p>Weights, gradients, m and v together: <b>16 bytes per parameter</b> before a single activation. For big models, the optimizer state is a major share of training memory.</p>' },
        { t: 'num', q: 'A model has 7 billion parameters. Adam keeps m and v for each one in fp32 (4 bytes each). How many GB is that optimizer state?', a: 56, tol: 1.12, ex: '7 billion × 2 values × 4 bytes = 56 GB, twice the size of the fp32 weights themselves.', gen: gen.adamMem },
        { t: 'num', q: '7 billion parameters, all fp32. Counting the weights, their gradients, and Adam’s m and v, how many GB before any activations?', a: 112, tol: 2.24, ex: '4 copies × 4 bytes = 16 bytes per parameter: 7 billion × 16 = 112 GB. Adam’s state is half of it.', gen: gen.adamTotal },
        { t: 'mc', q: 'Which statement about Adam’s state is right?', o: ['m and v each hold one value per parameter, so they cost twice the weights’ memory', 'm and v are single numbers shared by all parameters', 'Adam stores only the last gradient', 'Adam’s state is freed after every step'], ex: 'Adam’s per-parameter scaling is exactly what costs memory: it needs per-parameter statistics, kept for the whole of training.' },
        { t: 'explain', kind: 'recall', secs: 45, q: 'Explain Adam in 45 seconds: what are m and v, and why does Adam cost memory?', model: '<p>Adam keeps two moving averages per parameter. <b>m</b> averages the gradients: the direction. <b>v</b> averages the squared gradients: the scale. Each step moves by about η·m̂/√v̂, so every parameter gets a normalized step whose size barely depends on how big its gradients are. The cost is memory: m and v are one value each per parameter, so the optimizer state is twice the size of the weights.</p>', rubric: ['m is a moving average of gradients (direction)', 'v is a moving average of squared gradients (scale)', 'The step is about η·m̂/√v̂: normalized per parameter', 'Two extra values per parameter: memory twice the weights'] },
      ],
      cards: [
        { id: 'bp-adam-m', front: 'Adam’s first moment tracks what?', back: 'The direction of recent gradients.' },
        { id: 'bp-adam-v', front: 'Adam’s second moment tracks what?', back: 'The size of recent gradients.' },
        { id: 'bp-adam-hook', front: 'How long does Adam keep its m and v?', back: 'For the whole of training.' },
        { id: 'bp-adam-memory', front: 'Why does Adam cost so much memory?', back: 'It keeps two extra values per parameter.' },
        { id: 'bp-adam-scale', front: 'Do bigger gradients mean bigger Adam steps?', back: 'Barely: Adam normalizes by the gradient’s scale.' },
      ],
    },
  ],
};
