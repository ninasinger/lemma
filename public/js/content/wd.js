// Unit: weight decay. Numbers in wd-2 come from scripts/verify_numpy.py (and match the optim widget).

import { num, big, shape, py } from '../vary.js';
import { sup } from '../util.js';

// Fresh numbers every time a question or card is shown (see ../vary.js).
const LR = { 0.003: '3×10⁻³', 0.001: '10⁻³', 0.0003: '3×10⁻⁴', 0.0001: '10⁻⁴' };
const IN = [784, 512, 256, 100, 20], OUT = [128, 64, 32, 10, 3];
const gen = {
  oneStep(v) {
    const w = v.pick([1, 2, 3, 0.5, -2, -1.5]), eta = v.pick([0.1, 0.01, 0.5]), lam = v.pick([0.1, 0.2, 0.5, 1]), f = 1 - eta * lam, a = +(f * w).toFixed(6);
    return { q: `w = ${num(w)}, learning rate η = ${num(eta)}, decay λ = ${num(lam)}, and the gradient is 0. What is w after one step?`, a,
      ex: `(1 − ${num(eta)} × ${num(lam)}) × ${num(w)} = ${num(f)} × ${num(w)} = ${num(a)}.`, py: `w = ${w}\nw = w - ${eta} * (0 + ${lam} * w)\nans = w` };
  },
  manySteps(v) {
    let el, t;
    do { el = v.pick([0.001, 0.002, 0.005, 0.01, 0.02]); t = v.pick([50, 100, 200, 500, 1000]); } while (el * t < 0.2 || el * t > 3);
    const a = +((1 - el) ** t).toFixed(3);
    return { q: `ηλ = ${num(el)} and the gradient is 0. After ${big(t)} steps, the weight has been multiplied by about…`, a,
      ex: `${num(1 - el)}${sup(t)} ≈ ${num(a, 3)}, very close to exp(−${num(el * t)}) ≈ ${num(Math.exp(-el * t), 3)}.`,
      py: `w = 1.0\nfor _ in range(${t}):\n    w *= 1 - ${el}\nans = w` };
  },
  adamw(v) {
    let lam, lr, T;
    do { lam = v.pick([0.01, 0.05, 0.1]); lr = v.pick([0.003, 0.001, 0.0003, 0.0001]); T = v.pick([1000, 2000, 5000, 10000]); } while (lam * lr * T < 0.02 || lam * lr * T > 1.5);
    const x = lam * lr * T, exact = (1 - lr * lam) ** T;
    return { q: `PyTorch AdamW with λ = ${num(lam)} runs ${big(T)} steps at a constant lr = ${LR[lr]}. By about what factor has an unused weight been multiplied?`, a: +exact.toFixed(3),
      ex: `exp(−λ·Σlr) = exp(−${num(lam)} × ${num(lr)} × ${big(T)}) = exp(−${num(x)}) ≈ ${num(Math.exp(-x), 3)}. Exactly, (1 − ${num(lr * lam, 8)})${sup(T)} = ${num(exact, 4)}.`,
      py: `w = 1.0\nfor _ in range(${T}):\n    w = w * (1 - ${lr} * ${lam})\nans = w` };
  },
  layerOut(v) {
    const b = v.pick([16, 32, 64, 128]), i = v.pick(IN), o = v.pick(OUT);
    return { code: `x = torch.zeros(${b}, ${i})\nlayer = nn.Linear(${i}, ${o})\ny = layer(x)`, a: shape(b, o),
      ex: `(${b}, ${i}) @ (${i}, ${o}) → (${b}, ${o}), plus the bias broadcast across the ${b} rows.`,
      py: `x, W, bias = np.zeros((${b}, ${i})), np.zeros((${o}, ${i})), np.zeros(${o})\nans = (x @ W.T + bias).shape` };
  },
  params(v) {
    const i = v.pick(IN), o = v.pick(OUT), a = i * o + o;
    return { q: `How many parameters does <code>nn.Linear(${i}, ${o})</code> have, including the bias?`, a,
      ex: `${i} · ${o} weights + ${o} biases = ${big(i * o)} + ${o} = ${big(a)}.`, py: `ans = np.zeros((${o}, ${i})).size + np.zeros(${o}).size` };
  },
  gradShape(v) {
    const i = v.pick(IN), o = v.pick(OUT);
    return { code: `layer = nn.Linear(${i}, ${o})\nloss.backward()\nlayer.weight.grad`, a: shape(o, i),
      py: `x, dy = np.zeros((4, ${i})), np.ones((4, ${o}))\nans = (dy.T @ x).shape` };
  },

};

export default {
  id: 'wd',
  title: 'Weight decay',
  sub: 'Shrink every weight a little, every single step.',
  c: 'wd',
  guide: `<div class="eq">w ← (1 − ηλ)·w − η·∇L</div><ul><li>Every step, each weight loses the fraction ηλ of itself. With zero gradient: wₜ = (1 − ηλ)ᵗ·w₀.</li><li>Equivalent to L2 with penalty (λ/2)‖w‖² <b>only for plain SGD</b>. Momentum, Adam and most variants break it.</li><li>PyTorch <code>SGD(momentum, weight_decay)</code> and <code>Adam(weight_decay)</code> are L2: λw is added to the gradient. <code>AdamW(weight_decay)</code> is decoupled.</li><li>In PyTorch’s AdamW each step multiplies w by (1 − lr·λ): total shrink ≈ exp(−λ Σ lr_t), so the learning-rate schedule changes the regularization.</li><li>Parameters with <code>grad=None</code> are skipped entirely, decay included.</li><li>Skipping decay for biases and norm gains is a convention, not a law.</li></ul>
  <h3>Adam</h3><div class="eq">m ← β₁m + (1 − β₁)g<br>v ← β₂v + (1 − β₂)g²<br>w ← w − η·m̂ / (√v̂ + ε)</div><p class="note">m̂ = m/(1 − β₁ᵗ), v̂ = v/(1 − β₂ᵗ). Adam + L2 uses g = ∇L + λw in both moments; AdamW uses g = ∇L and adds w ← w(1 − ηλ).</p>
  <h3>Shapes</h3><pre class="py">nn.Linear(in, out).weight   # (out, in)
nn.Linear(in, out).bias     # (out,)
layer(x)                    # x @ W.T + b: (batch, in) -> (batch, out)
p.grad.shape == p.shape     # always (or p.grad is None)</pre>`,
  lessons: [
    {
      id: 'wd-1',
      title: 'Shrink every step',
      steps: [
        { t: 'learn', title: 'A slow leak in every weight', body: `<p>Weight decay changes the update rule itself. Each step, every weight loses a small fraction of itself on top of following the gradient:</p><div class="eq">w ← w − η·∇L − η·λ·w<br>  = (1 − ηλ)·w − η·∇L</div><p>Here only w₁ matters to the loss; w₂–w₅ only see noise. Press Run with decay off, then reset and try it on.</p>`, w: 'decay' },
        { t: 'num', q: 'w = 2.0, learning rate η = 0.1, decay λ = 0.5, and the gradient is 0. What is w after one step?', a: 1.9, tol: 0.001, ex: '(1 − 0.1 × 0.5) × 2.0 = 0.95 × 2.0 = 1.9.', gen: gen.oneStep },
        { t: 'task', title: 'Clean up the noise', body: '<p>Turn weight decay on and train until every useless weight (w₂ to w₅) is smaller than <b>0.05</b> in size.</p>', w: 'decay', hint: 'Weight decay has to be on. Then press Run (or Step many times) and watch w₂–w₅ fade.', ex: 'The gradient keeps w₁ alive, so it settles at 1.2 / (1 + λ) instead of 1.2. Nothing defends the other weights, so they fade to zero.' },
        { t: 'mc', q: 'With no gradient signal at all, weight decay makes a weight…', o: ['shrink exponentially toward 0', 'jump straight to 0', 'shrink by the same amount every step', 'oscillate around 0'], ex: 'Each step multiplies by (1 − ηλ), so after t steps w = (1 − ηλ)ᵗ·w₀. That is exponential decay, hence the name.' },
        { t: 'num', q: 'ηλ = 0.01 and the gradient is 0. After 100 steps, the weight has been multiplied by about…', a: 0.366, tol: 0.01, ex: '0.99¹⁰⁰ ≈ 0.366, very close to e⁻¹ ≈ 0.368.', gen: gen.manySteps },
      ],
      cards: [
        { id: 'wd-update', front: 'What does weight decay do each step?', back: 'Shrinks every weight by a small fraction.' },
        { id: 'wd-exp', front: 'With no gradient, what happens to a decayed weight over time?', back: 'It shrinks exponentially toward zero.' },
        { id: 'wd-unused', front: 'Under weight decay, what happens to weights the loss doesn’t need?', back: 'They decay to zero.' },
      ],
    },
    {
      id: 'wd-2',
      title: 'Adam, L2 and AdamW, in full',
      steps: [
        { t: 'learn', title: 'Twins only under plain SGD', body: `<p>Add the penalty <span class="m">(λ/2)‖w‖²</span> to the loss. Its gradient is <span class="m">λw</span>, so plain SGD gives</p><div class="eq">w ← w − η·(∇L + λw) = (1 − ηλ)·w − η·∇L</div><p>That is exactly weight decay. But the equivalence belongs to this one update rule. Anything that transforms the gradient before applying it (momentum, Adam, gradient clipping) treats the λw term differently from a direct shrink.</p>` },
        { t: 'learn', title: 'Momentum already breaks it', body: `<p>PyTorch’s <code>SGD(momentum=0.9, weight_decay=λ)</code> adds λw to the gradient, so the penalty flows through the momentum buffer:</p><div class="eq">b ← μb + (∇L + λw),   w ← w − η·b</div><p>At steady state the buffer amplifies whatever goes into it by about <span class="m">1/(1 − μ)</span>, which is 10× for μ = 0.9, penalty included. Decoupled decay keeps it out of the buffer:</p><div class="eq">b ← μb + ∇L,   w ← w − η·b − η·λ·w</div>` },
        { t: 'learn', title: 'Adam, written out', body: `<div class="eq">m ← β₁m + (1 − β₁)·g<br>v ← β₂v + (1 − β₂)·g²<br>m̂ = m / (1 − β₁ᵗ),   v̂ = v / (1 − β₂ᵗ)<br>w ← w − η·m̂ / (√v̂ + ε)</div><p>m tracks the average gradient and v the average squared gradient, per coordinate. The hats undo the bias from starting both at zero, and ε (about 10⁻⁸) avoids dividing by zero. Each step is roughly <span class="m">η × average gradient / its typical size</span>, so steps are about η in size whatever the gradient’s scale.</p><ul><li><b>Adam + L2</b> (PyTorch <code>Adam(weight_decay=λ)</code>): g = ∇L + λw, so the penalty enters <em>both</em> m and v and is divided by √v̂.</li><li><b>AdamW</b> (PyTorch <code>AdamW(weight_decay=λ)</code>): g = ∇L only, plus a separate shrink w ← w(1 − ηλ).</li></ul>` },
        { t: 'explain', kind: 'derive', secs: 120, q: 'Two weights start at w = [1, 1] with data gradients g = [1, 0]. With η = 0.01, λ = 0.1 and ε ≈ 0, compute the first update (t = 1) under Adam + L2 and under AdamW, and explain why they differ.', model: `<p>On step 1, m̂ = g and v̂ = g², so each coordinate moves by <span class="m">η·g/|g| = η·sign(g)</span> (and by 0 where g = 0).</p><p><b>Adam + L2:</b> g = ∇L + λw = [1.1, 0.1]. Both are positive, so both weights move by 0.01: <b>w = [0.99, 0.99]</b>. The unused weight shrinks as much as the busy one moves; the normalization erased the penalty’s size.</p><p><b>AdamW:</b> first w ← w(1 − ηλ) = [0.999, 0.999], then the Adam step on g = [1, 0] subtracts [0.01, 0]: <b>w = [0.989, 0.999]</b>.</p><p>The difference: Adam + L2 lets the per-coordinate normalization rescale the penalty, while AdamW shrinks every weight by the same factor, 1 − ηλ.</p>`, rubric: ['States that Adam’s first step is η·sign(g) per coordinate', 'Adam + L2 gives [0.99, 0.99]', 'AdamW gives [0.989, 0.999]', 'Explains that normalization rescales the L2 term per coordinate', 'Notes that AdamW shrinks every weight by the same factor 1 − ηλ'] },
        { t: 'learn', title: 'Two hundred steps later', body: '<p>Same η = 0.01 and λ = 0.1. First a weight the loss never uses, so only regularization moves it. Then switch to a busy weight that the loss pulls toward 2.</p>', w: 'optim' },
        { t: 'mc', q: 'After 200 steps the unused weight is 0.016 under Adam + L2 but 0.819 under AdamW. Why?', o: ['Adam normalizes λw into a full-size step of about η per step', 'AdamW skips decay for unused weights', 'Adam + L2 uses a larger learning rate', 'Momentum cancels the decay in AdamW'], ex: 'With no data gradient, λw is the whole gradient, and Adam turns it into steps of roughly η, so the weight falls almost linearly. AdamW shrinks it by exactly (1 − ηλ) per step: 0.999²⁰⁰ ≈ 0.819.' },
        { t: 'learn', title: 'Same λ, different regularization', body: `<div class="tblwrap"><table class="tbl"><thead><tr><th>Update rule</th><th>Unused weight after 200 steps</th><th>Busy weight, converged</th></tr></thead><tbody><tr><td>SGD + L2</td><td class="n">0.819</td><td class="n">1.818</td></tr><tr><td>SGD momentum + L2</td><td class="n">0.120</td><td class="n">1.818</td></tr><tr><td>SGD momentum, decoupled</td><td class="n">0.819</td><td class="n">1.980</td></tr><tr><td>Adam + L2</td><td class="n">0.016</td><td class="n">1.818</td></tr><tr><td>AdamW</td><td class="n">0.819</td><td class="n">≈ 1.98</td></tr></tbody></table></div><p>Every rule that adds λw to the gradient settles at the minimizer of L + (λ/2)‖w‖², here 2/(1 + λ) = 1.818, but by very different paths. The decoupled rules settle elsewhere: decoupled momentum behaves like L2 with λ scaled down by (1 − μ), and AdamW’s balance point depends on Adam’s per-coordinate scaling.</p><p><b>The value of λ isn’t portable across optimizers.</b> Retune it whenever the optimizer changes.</p>` },
        { t: 'learn', title: 'The learning rate is part of the decay', body: `<p>PyTorch’s AdamW multiplies each weight by <span class="m">(1 − lr·λ)</span> every step, so over training</p><div class="eq">total shrink = Π (1 − lr_t·λ) ≈ exp(−λ · Σ lr_t)</div><p>With λ = 0.1 and 1,000 steps at lr = 10⁻³ that’s about e^−0.1 = 0.905. A cosine schedule from 10⁻³ down to 0 has half the area, so about e^−0.05 = 0.951. <b>Changing the learning rate or its schedule changes the regularization</b>, even with λ fixed.</p><p>The AdamW paper scales decay by the schedule multiplier but not by the base learning rate; libraries differ, so check yours.</p>` },
        { t: 'num', q: 'PyTorch AdamW with λ = 0.05 runs 2,000 steps at a constant lr = 10⁻³. By about what factor has an unused weight been multiplied?', a: 0.905, tol: 0.003, ex: 'exp(−λ·Σlr) = exp(−0.05 × 0.001 × 2000) = e^−0.1 ≈ 0.905. Exactly, (1 − 5×10⁻⁵)²⁰⁰⁰ = 0.9048.', gen: gen.adamw },
        { t: 'tf', q: 'With plain SGD (no momentum), L2 regularization and weight decay give the same update when λ is matched.', a: true, ex: 'The penalty’s gradient λw is exactly the decay term. Add momentum or an adaptive optimizer and they diverge.' },
        { t: 'explain', kind: 'diagnose', secs: 75, q: 'You halved the learning rate of an AdamW run, changed nothing else, and the model now overfits more. Why might that be?', model: `<p>In PyTorch’s AdamW the per-step shrink is <span class="m">lr × weight_decay</span>, so halving the learning rate also halves the effective weight decay: the total shrink ≈ exp(−λ Σ lr_t) is weaker. Less regularization can mean more overfitting even though λ didn’t change.</p><p>To separate the effects, rerun with weight_decay doubled so lr × λ stays constant. Or tune them together. A smaller learning rate also changes gradient noise and how far training gets, so the decay isn’t the only suspect.</p>`, rubric: ['Per-step decay in PyTorch AdamW is lr × weight_decay', 'Total shrink depends on the sum of learning rates over the schedule', 'Halving lr halves the effective regularization', 'Proposes holding lr·λ fixed (or a joint sweep) to test it', 'Acknowledges other effects of a smaller learning rate'] },
        { t: 'match', q: 'Match each PyTorch setting to what it does.', p: [['Adam(weight_decay=λ)', 'Adds λw to the gradient (L2)'], ['AdamW(weight_decay=λ)', 'Shrinks w by lr·λ each step'], ['SGD(momentum=0.9, weight_decay=λ)', 'L2 through the momentum buffer'], ['Σ lr_t over training', 'Sets the total decay'] ], ex: 'The same keyword argument means different things in different optimizers.' },
      ],
      cards: [
        { id: 'wd-sgd', front: 'When are L2 regularization and weight decay the same?', back: 'Only for plain SGD.' },
        { id: 'wd-mom', front: 'In PyTorch’s SGD with momentum, does weight_decay act like L2?', back: 'Yes: it flows through the momentum buffer.' },
        { id: 'wd-adam', front: 'In Adam with L2, what happens to the penalty?', back: 'Adam rescales it along with the gradient.' },
        { id: 'wd-adamw', front: 'What does AdamW change about weight decay?', back: 'It applies the decay directly, outside the adaptive step.' },
        { id: 'wd-first-step', front: 'Adam’s first step depends on what about the gradient?', back: 'Only its sign.' },
        { id: 'wd-lr-coupling', front: 'In PyTorch’s AdamW, does the learning rate affect the decay?', back: 'Yes: halving it halves the decay.' },
        { id: 'wd-portable', front: 'Can you reuse a tuned weight decay after switching optimizers?', back: 'No: retune it.' },
        { id: 'wd-derive-two-updates', front: 'Which decays every weight at the same rate: Adam with L2 or AdamW?', back: 'AdamW.' },
        { id: 'wd-defend-sweep', front: 'In an AdamW sweep, which knob couples with weight decay?', back: 'The learning rate.' },
      ],
    },
    {
      id: 'wd-3',
      title: 'Shapes in a PyTorch layer',
      steps: [
        { t: 'learn', title: 'nn.Linear stores (out, in)', body: `<pre class="py">layer = nn.Linear(in_features=784, out_features=128)
layer.weight.shape   # (128, 784)   (out, in), not (in, out)!
layer.bias.shape     # (128,)
x.shape              # (32, 784)    a batch of 32 inputs
y = layer(x)         # x @ layer.weight.T + layer.bias
y.shape              # (32, 128)</pre><p>Each <b>row</b> of the weight matrix holds the 784 incoming weights of one output unit.</p>` },
        { t: 'shape', q: 'What is the shape of <code>y</code>?', code: 'x = torch.zeros(64, 784)\nlayer = nn.Linear(784, 128)\ny = layer(x)', a: '(64, 128)', ex: '(64, 784) @ (784, 128) → (64, 128), plus the bias broadcast across the 64 rows.', gen: gen.layerOut },
        { t: 'mc', q: 'What is <code>layer.weight[j]</code> (row j)?', o: ['The 784 incoming weights of output unit j', 'The weights leaving input j', 'The bias of unit j', 'One input example'], ex: 'y[:, j] = x @ weight[j] + bias[j]: row j is dotted with each input to produce output unit j.' },
        { t: 'num', q: 'How many parameters does <code>nn.Linear(784, 128)</code> have, including the bias?', a: 100480, tol: 0, ex: '784 · 128 weights + 128 biases = 100,352 + 128 = 100,480.', gen: gen.params },
        { t: 'learn', title: 'One step of decoupled decay', body: `<pre class="py">with torch.no_grad():
    for p in model.parameters():
        if p.grad is None:
            continue           # not used this step: no update, no decay
        p.mul_(1 - lr * wd)    # decay: elementwise, same shape as p
        p.sub_(lr * p.grad)    # gradient step: p.grad has p's shape</pre><p>Decay multiplies every entry of the (128, 784) matrix by the same 1 − ηλ. Nothing about the shape changes.</p>` },
        { t: 'shape', q: 'What is the shape of the weight’s gradient?', code: 'layer = nn.Linear(784, 128)\nloss.backward()\nlayer.weight.grad', a: '(128, 784)', ex: 'A gradient always has exactly its parameter’s shape: one partial derivative per entry.', gen: gen.gradShape },
        { t: 'learn', title: 'None is not zero', body: '<p>With <code>optimizer.zero_grad()</code> (which sets gradients to <code>None</code> by default), a parameter that takes no part in the forward pass keeps <code>grad = None</code>. PyTorch’s optimizers <b>skip such parameters entirely</b>: no gradient step and no weight decay, not even AdamW’s decoupled decay.</p><p>A gradient of zeros is different. The parameter is processed: weight decay applies, and Adam’s moment estimates keep decaying, so the parameter can even move.</p>' },
        { t: 'tf', q: 'AdamW applies weight decay to every parameter in its param groups on every step.', a: false, ex: 'Parameters whose <code>.grad</code> is <code>None</code> are skipped, decay included. Unused embedding rows and branches you didn’t run this step are typical examples.' },
        { t: 'learn', title: 'Choosing what to decay', body: `<pre class="py">decay    = [p for p in model.parameters() if p.ndim >= 2]   # weight matrices
no_decay = [p for p in model.parameters() if p.ndim &lt; 2]    # biases, norm gains
opt = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=3e-4)</pre><p>The shape does the sorting: weight matrices have 2 dimensions, while biases and LayerNorm gains are 1-D. It’s a <b>heuristic</b>, though. Embeddings are 2-D and some setups exclude them, and nothing says every 1-D parameter should be left alone. Treat the grouping as a modeling choice you can test.</p>` },
        { t: 'mc', q: 'Why does <code>p.ndim &gt;= 2</code> usually pick out the parameters to decay?', o: ['Weight matrices are 2-D; biases and norm gains are 1-D', '2-D parameters train faster', 'PyTorch requires it', 'It skips the embeddings'], ex: 'It’s a shape-based shortcut for a common convention, not a rule. Check whether it matches what you intend for embeddings and any special parameters.' },
      ],
      cards: [
        { id: 'wd-np-linear', front: 'How is nn.Linear’s weight laid out?', back: 'Outputs by inputs.' },
        { id: 'wd-np-row', front: 'What is one row of a Linear layer’s weight?', back: 'One output unit’s incoming weights.' },
        { id: 'wd-np-grad', front: 'What shape is a parameter’s gradient?', back: 'The same as the parameter.' },
        { id: 'wd-np-none', front: 'How do optimizers treat grad=None?', back: 'They skip the parameter entirely.' },
        { id: 'wd-nodecay', front: 'Which parameters usually skip weight decay?', back: 'Biases and normalization gains.' },
      ],
    },
  ],
};
