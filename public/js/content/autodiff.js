// Unit: chain rule review. No new theory: the backprop procedure from the Derivatives unit,
// drilled on near-identical graphs (each with fresh numbers) until it runs without thinking.
import { num, term } from '../vary.js';

// Fresh numbers every time a question or card is shown (see ../vary.js). Each py recomputes the
// answer with deriv(), the complex-step derivative in scripts/verify_numpy.py.
const plus = (k) => (k < 0 ? `− ${-k}` : `+ ${k}`);

// The review's main graph: a = p·x, b = a², c = q·b, y = c + r·x.
// Indirect path y ← c ← b ← a ← x, plus the direct edge y ← x.
function graph(v) {
  const p = v.pick([2, 3]), x = v.pick([-2, -1, 1, 2, 3]), q = v.int(2, 5), r = v.nz(-4, 6);
  const a = p * x, b = a * a, c = q * b;
  return { p, x, q, r, a, b, c, y: c + r * x, ind: q * 2 * a * p };
}
const show = (G) => `x = ${num(G.x)}, a = ${G.p}x, b = a², c = ${G.q}b, y = c ${plus(G.r)}x`;
const fy = (G) => `lambda x: ${G.q} * (${G.p} * x)**2 + ${G.r} * x`;
const forward = (G) => `a = ${num(G.a)}, b = ${num(G.b)}, c = ${num(G.c)}, y = ${num(G.y)}`;
const edges = (G) => `1 × ${G.q} × ${term(2 * G.a)} × ${G.p}`;

// One local rule, picked at random, with an incoming gradient that is usually not 1.
function local(v) {
  const g = v.pick([2, 3, 4, 5, -2, -3, 0.5]), kind = v.int(0, 3);
  if (kind === 0) {
    const a = v.nz(-5, 6), b = v.nz(-5, 6), toA = v.int(0, 1) === 1, l = toA ? b : a;
    return { q: `c = a·b with a = ${num(a)}, b = ${num(b)}, and ∂L/∂c = ${num(g)}. What gradient does ${toA ? 'a' : 'b'} receive?`, a: +(g * l).toFixed(6),
      ex: `Local ∂c/∂${toA ? 'a' : 'b'} = ${toA ? 'b' : 'a'} = ${num(l)} (the other input, not its own value). Incoming × local: ${num(g)} × ${term(l)} = ${num(g * l)}.`,
      py: `ans = ${g} * deriv(lambda a, b: a * b, ${a}, ${b}, wrt=${toA ? 0 : 1})` };
  }
  if (kind === 1) {
    const a = v.nz(-9, 9), b = v.nz(-9, 9);
    return { q: `c = a + b with a = ${num(a)}, b = ${num(b)}, and ∂L/∂c = ${num(g)}. What gradient does a receive?`, a: g,
      ex: `Local ∂c/∂a = 1, whatever the values. Incoming × local: ${num(g)} × 1 = ${num(g)}.`, py: `ans = ${g} * deriv(lambda a, b: a + b, ${a}, ${b}, wrt=0)` };
  }
  if (kind === 2) {
    const x = v.nz(-4, 5);
    return { q: `c = x² with x = ${num(x)}, and ∂L/∂c = ${num(g)}. What gradient is sent to x?`, a: +(g * 2 * x).toFixed(6),
      ex: `Local dc/dx = 2x = ${num(2 * x)}. Incoming × local: ${num(g)} × ${term(2 * x)} = ${num(g * 2 * x)}.`, py: `ans = ${g} * deriv(lambda x: x**2, ${x})` };
  }
  const k = v.pick([2, 3, 4, -3, 10]);
  return { q: `c = ${num(k)}a and ∂L/∂c = ${num(g)}. What gradient does a receive?`, a: +(g * k).toFixed(6),
    ex: `Local ∂c/∂a = ${num(k)}. Incoming × local: ${num(g)} × ${term(k)} = ${num(g * k)}.`, py: `ans = ${g} * deriv(lambda a: ${k} * a, 1.0)` };
}

const gen = {
  value(v) {
    const G = graph(v), node = v.pick(['b', 'c', 'y']);
    return { q: `${show(G)}. What is the value of ${node}?`, a: G[node], ex: `Forward pass, one node at a time: ${forward(G)}.`,
      py: `x = ${G.x}\na = ${G.p} * x\nb = a**2\nc = ${G.q} * b\ny = c + ${G.r} * x\nans = ${node}` };
  },
  edge(v) {
    const G = graph(v);
    return { q: `${show(G)}. What is the local derivative ∂b/∂a?`, a: 2 * G.a,
      ex: `b = a², so ∂b/∂a = 2a = 2 × ${term(G.a)} = ${num(2 * G.a)}. Not ${num(G.b)}: that is b’s value.`, py: `ans = deriv(lambda a: a**2, ${G.a})` };
  },
  indirect(v) {
    const G = graph(v);
    return { q: `${show(G)}. What does the path x → a → b → c → y contribute to ∂y/∂x?`, a: G.ind,
      ex: `Values: a = ${num(G.a)}. Edges: ∂y/∂c = 1, ∂c/∂b = ${G.q}, ∂b/∂a = 2a = ${num(2 * G.a)}, ∂a/∂x = ${G.p}. Multiply along the path: ${edges(G)} = ${num(G.ind)}.`,
      py: `x0 = ${G.x}\nans = deriv(lambda x: ${G.q} * (${G.p} * x)**2 + ${G.r} * x0, ${G.x})` };
  },
  direct(v) {
    const G = graph(v);
    return { q: `${show(G)}. What does the direct edge x → y contribute to ∂y/∂x?`, a: G.r,
      ex: `y = c ${plus(G.r)}x, so the direct edge carries ∂y/∂x = ${num(G.r)}.`, py: `x0 = ${G.x}\nans = deriv(lambda x: ${G.q} * (${G.p} * x0)**2 + ${G.r} * x, ${G.x})` };
  },
  total(v) {
    const G = graph(v);
    return { q: `${show(G)}. What is ∂y/∂x?`, a: G.r + G.ind,
      ex: `Forward: ${forward(G)}. Indirect path: ${edges(G)} = ${num(G.ind)}. Direct edge: ${num(G.r)}. Add across paths: ${num(G.ind)} + ${term(G.r)} = ${num(G.r + G.ind)}.`,
      py: `ans = deriv(${fy(G)}, ${G.x})` };
  },
  totalIncoming(v) {
    const G = graph(v), g = v.pick([2, 3, -1, -2, 0.5, 10]), t = G.r + G.ind;
    return { q: `${show(G)}, and y feeds the loss with ∂L/∂y = ${num(g)}. What is ∂L/∂x?`, a: +(g * t).toFixed(6),
      ex: `∂y/∂x = ${num(G.ind)} (indirect) + ${term(G.r)} (direct) = ${num(t)}. The incoming gradient isn’t 1, so multiply it in: ${num(g)} × ${term(t)} = ${num(g * t)}.`,
      py: `ans = ${g} * deriv(${fy(G)}, ${G.x})` };
  },
  shifted(v) {
    const p = v.nz(-3, 3), q = v.pick([2, 3, -2]), r = v.nz(-4, 5), x = v.pick([-2, -1, 1, 2]);
    const a = x + p, b = q * a, ind = 2 * b * q;
    return { q: `x = ${num(x)}, a = x ${plus(p)}, b = ${num(q)}a, c = b², y = c ${plus(r)}x. What is ∂y/∂x?`, a: ind + r,
      ex: `Forward: a = ${num(a)}, b = ${num(b)}. Indirect path: ∂y/∂c · ∂c/∂b · ∂b/∂a · ∂a/∂x = 1 × ${term(2 * b)} × ${term(q)} × 1 = ${num(ind)}. Direct: ${num(r)}. Add: ${num(ind + r)}.`,
      py: `ans = deriv(lambda x: (${q} * (x + ${p}))**2 + ${r} * x, ${x})` };
  },

  local,
  valueTrap(v) {
    const a = v.nz(-5, 9), b = v.nz(-5, 9);
    return { q: `y = a·b with a = ${num(a)} and b = ${num(b)}. What is ∂y/∂a?`, a: b,
      ex: `∂y/∂a = b = ${num(b)}: the other input’s value. a’s own value (${num(a)}) is not its derivative.`, py: `ans = deriv(lambda a, b: a * b, ${a}, ${b}, wrt=0)` };
  },

  chain3(v) {
    const p = v.int(2, 4), k = v.nz(-3, 3), x = v.nz(-2, 3), a = p * x, b = a + k;
    return { q: `a = ${p}x, b = a ${plus(k)}, y = b², at x = ${num(x)}. What is dy/dx?`, a: 2 * b * p,
      ex: `Values: a = ${num(a)}, b = ${num(b)}. Edges: dy/db = 2b = ${num(2 * b)}, db/da = 1, da/dx = ${p}. Multiply: ${term(2 * b)} × 1 × ${p} = ${num(2 * b * p)}. Skipping b and writing 2a would give the wrong answer.`,
      py: `ans = deriv(lambda x: (${p} * x + ${k})**2, ${x})` };
  },
  indirectOnly(v) {
    const c = v.int(2, 5), k = v.nz(-6, 6), m = v.nz(-4, 5), x = v.int(1, 9);
    return { q: `a = ${c}x, b = a ${plus(k)}, y = ${num(m)}b, at x = ${x}. x is not a parent of y. What is ∂y/∂x?`, a: c * m,
      ex: `No direct edge, but a path: ${c} × 1 × ${term(m)} = ${num(c * m)}. Indirect influence still counts.`, py: `ans = deriv(lambda x: ${m} * (${c} * x + ${k}), ${x})` };
  },
  branchSum(v) {
    const p = v.int(2, 6), x = v.nz(-4, 5);
    return { q: `a = ${p}x, b = x², y = a + b, at x = ${num(x)}. What is dy/dx?`, a: p + 2 * x,
      ex: `Path through a: 1 × ${p} = ${p}. Path through b: 1 × 2x = ${num(2 * x)}. Add across paths: ${p} + ${term(2 * x)} = ${num(p + 2 * x)}.`, py: `ans = deriv(lambda x: ${p} * x + x**2, ${x})` };
  },
  branchProduct(v) {
    const p = v.int(2, 5), q = v.int(2, 5), x = v.nz(-3, 3), a = p * x, b = q * x;
    return { q: `a = ${p}x, b = ${q}x, y = a·b, at x = ${num(x)}. What is dy/dx?`, a: b * p + a * q,
      ex: `Values: a = ${num(a)}, b = ${num(b)}. Through a: ∂y/∂a · da/dx = b · ${p} = ${num(b * p)}. Through b: a · ${q} = ${num(a * q)}. Add: ${num(b * p + a * q)}.`,
      py: `ans = deriv(lambda x: (${p} * x) * (${q} * x), ${x})` };
  },
  accumulate(v) {
    const u = v.nz(-9, 12), w = v.nz(-9, 12);
    return { q: `x reaches y along two paths. One path’s product is ${num(u)}, the other’s is ${num(w)}. What is ∂y/∂x?`, a: u + w,
      ex: `Accumulate means add: ${num(u)} + ${term(w)} = ${num(u + w)}.`, py: `ans = deriv(lambda x: ${u} * x + ${w} * x, 0.0)` };
  },
  branchIncoming(v) {
    const p = v.int(2, 5), x = v.nz(-3, 4), g = v.pick([2, 3, -2, 0.5, 4]), t = p + 2 * x;
    return { q: `a = ${p}x, b = x², y = a + b, at x = ${num(x)}, with ∂L/∂y = ${num(g)}. What is ∂L/∂x?`, a: +(g * t).toFixed(6),
      ex: `dy/dx = ${p} + ${term(2 * x)} = ${num(t)} (add across paths). Incoming isn’t 1: ${num(g)} × ${term(t)} = ${num(g * t)}.`,
      py: `ans = ${g} * deriv(lambda x: ${p} * x + x**2, ${x})` };
  },

  cTotal(v) {
    const G = graph(v);
    return { front: `${show(G)}. ∂y/∂x?`, back: `${edges(G)} + ${term(G.r)} = <b>${num(G.r + G.ind)}</b>.`, a: G.r + G.ind, py: `ans = deriv(${fy(G)}, ${G.x})` };
  },
  cValueVsLocal(v) {
    const a = v.nz(-5, 6);
    return { front: `b = a², a = ${num(a)}. Value of b, and ∂b/∂a?`, back: `b = <b>${num(a * a)}</b>; ∂b/∂a = 2a = <b>${num(2 * a)}</b>. Different numbers.`, a: [a * a, 2 * a],
      py: `ans = [(${a})**2, deriv(lambda a: a**2, ${a})]` };
  },
  cLocal(v) {
    const L = local(v);
    return { front: L.q, back: `<b>${num(L.a)}</b>. ${L.ex}`, a: L.a, py: L.py };
  },
  cBranch(v) {
    const p = v.int(2, 6), x = v.nz(-4, 5);
    return { front: `a = ${p}x, b = x², y = a + b, x = ${num(x)}. dy/dx?`, back: `${p} + ${term(2 * x)} = <b>${num(p + 2 * x)}</b>.`, a: p + 2 * x, py: `ans = deriv(lambda x: ${p} * x + x**2, ${x})` };
  },
  cIncoming(v) {
    const G = graph(v), g = v.pick([2, -1, 0.5, 3]), t = G.r + G.ind;
    return { front: `${show(G)}, ∂L/∂y = ${num(g)}. ∂L/∂x?`, back: `${num(g)} × ${term(t)} = <b>${num(g * t)}</b>.`, a: +(g * t).toFixed(6), py: `ans = ${g} * deriv(${fy(G)}, ${G.x})` };
  },
};

export default {
  id: 'ad',
  title: 'Chain rule review',
  sub: 'The backprop procedure, drilled until it runs on its own.',
  c: 'dv',
  guide: `<h3>Memory hooks</h3><div class="eq hook">Incoming gradient × local derivative</div><div class="eq hook">Multiply along paths; add across paths</div><div class="eq hook">Forward = values. Backward = sensitivities.</div><div class="eq hook">One scalar loss, many parameters → reverse mode</div>
  <h3>Core idea</h3><ul><li>The forward pass computes values and records the computation graph.</li><li>The backward pass computes sensitivities (gradients).</li><li>Start from the scalar loss with ∂L/∂L = 1.</li></ul>
  <h3>Local derivative rule</h3><ul><li>Gradient sent to a parent = incoming gradient × local derivative.</li><li>y = a + b: local derivatives 1 and 1.</li><li>y = a·b: ∂y/∂a = b, ∂y/∂b = a.</li><li>y = x²: 2x.</li></ul>
  <h3>Path rule</h3><ul><li>Multiply along a path.</li><li>Add across paths.</li><li>A variable that reaches the output through several branches accumulates every contribution.</li></ul>
  <h3>Checklist</h3><ol><li>Write the full path first.</li><li>Compute the intermediate values.</li><li>Write one local derivative per edge.</li><li>Multiply along each path.</li><li>Add the contributions from separate paths.</li></ol>
  <h3>Common mistakes</h3><ul><li>Confusing a node’s value with its derivative.</li><li>Skipping an intermediate node.</li><li>Forgetting that a variable can influence the output indirectly.</li><li>Overwriting instead of accumulating: “accumulate” means add.</li><li>Forgetting the incoming gradient when it isn’t 1: multiply by it.</li></ul>
  <h3>Worked example</h3><p>x = 2, a = 2x, b = a², c = 4b, y = c + 3x.</p><ul><li>Values: a = 4, b = 16, c = 64, y = 67.</li><li>Direct path: ∂y/∂x = 3.</li><li>Indirect path: 1 × 4 × 8 × 2 = 64 (∂y/∂c, ∂c/∂b, ∂b/∂a = 2a, ∂a/∂x).</li><li>Total: 3 + 64 = 67.</li></ul>`,
  lessons: [
    {
      id: 'ad-1',
      title: 'One graph, step by step',
      steps: [
        { t: 'learn', title: 'The checklist', body: '<ol><li>Write the full path first.</li><li>Compute the intermediate values.</li><li>Write one local derivative per edge.</li><li>Multiply along each path.</li><li>Add the contributions from separate paths.</li></ol><p>Every drill below is the same graph shape with new numbers:</p><div class="eq">a = 2x, b = a², c = 4b, y = c + 3x</div>' },
        { t: 'learn', title: 'Worked once', body: '<p>x = 2. <b>Values:</b> a = 4, b = 16, c = 64, y = 67.</p><p><b>Indirect path</b> x → a → b → c → y: ∂y/∂c = 1, ∂c/∂b = 4, ∂b/∂a = 2a = 8, ∂a/∂x = 2. Multiply: 1 × 4 × 8 × 2 = 64.</p><p><b>Direct edge</b> x → y: 3.</p><p><b>Total:</b> 64 + 3 = 67.</p>' },
        { t: 'num', q: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x. What is the value of b?', a: 16, tol: 0, ex: 'Forward pass, one node at a time: a = 4, b = 16, c = 64, y = 67.', gen: gen.value },
        { t: 'num', q: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x. What is the local derivative ∂b/∂a?', a: 8, tol: 0, ex: 'b = a², so ∂b/∂a = 2a = 8. Not 16: that is b’s value.', gen: gen.edge },
        { t: 'num', q: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x. What does the path x → a → b → c → y contribute to ∂y/∂x?', a: 64, tol: 0, ex: 'Edges: 1, 4, 2a = 8, 2. Multiply along the path: 1 × 4 × 8 × 2 = 64.', gen: gen.indirect },
        { t: 'num', q: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x. What does the direct edge x → y contribute to ∂y/∂x?', a: 3, tol: 0, ex: 'y = c + 3x, so the direct edge carries 3.', gen: gen.direct },
        { t: 'num', q: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x. What is ∂y/∂x?', a: 67, tol: 0, ex: 'Indirect 64 plus direct 3: 67.', gen: gen.total },
        { t: 'mc', q: 'Someone gets ∂y/∂x = 3 for this graph. What did they miss?', o: ['The indirect path through a, b and c', 'The direct edge', 'The incoming gradient', 'Nothing: 3 is right'], ex: 'x is not a parent of c, but it reaches y through a → b → c. That path counts, and here it carries most of the gradient.' },
        { t: 'num', q: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x. What is ∂y/∂x?', a: 67, tol: 0, ex: 'Indirect 64 plus direct 3: 67.', gen: gen.total },
      ],
      cards: [
        { id: 'ad-checklist', front: 'The five-step checklist?', back: 'Write the paths. Compute values. One local derivative per edge. Multiply along each path. Add across paths.' },
        { id: 'ad-example', front: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x. ∂y/∂x?', back: '1 × 4 × 8 × 2 + 3 = <b>67</b>.', gen: gen.cTotal },
        { id: 'ad-value-local', front: 'b = a², a = 4. Value of b, and ∂b/∂a?', back: 'b = <b>16</b>; ∂b/∂a = 2a = <b>8</b>. Different numbers.', gen: gen.cValueVsLocal },
        { id: 'ad-miss-indirect', front: 'Answer came out as just the direct edge. What was missed?', back: 'The indirect path.' },
      ],
    },
    {
      id: 'ad-2',
      title: 'Local rules, fast',
      steps: [
        { t: 'learn', title: 'One rule per edge', body: '<div class="eq hook">gradient to parent = incoming gradient × local derivative</div><p>a + b → 1 and 1. a·b → b and a. x² → 2x. k·a → k.</p><p>If the incoming gradient isn’t 1, multiply by it.</p>' },
        { t: 'num', q: 'c = a·b with a = 2, b = 3, and ∂L/∂c = 5. What gradient does a receive?', a: 15, tol: 0, ex: 'Local ∂c/∂a = b = 3. Incoming × local: 5 × 3 = 15.', gen: gen.local },
        { t: 'num', q: 'c = a + b with a = 4, b = −1, and ∂L/∂c = 3. What gradient does a receive?', a: 3, tol: 0, ex: 'Local ∂c/∂a = 1. Incoming × local: 3 × 1 = 3.', gen: gen.local },
        { t: 'num', q: 'c = x² with x = 3, and ∂L/∂c = 2. What gradient is sent to x?', a: 12, tol: 0, ex: 'Local dc/dx = 2x = 6. Incoming × local: 2 × 6 = 12.', gen: gen.local },
        { t: 'mc', q: 'y = a·b with a = 5 and b = −2. Is ∂y/∂a equal to 5?', o: ['No: it’s b = −2', 'Yes', 'No: it’s 1', 'No: it’s −10'], ex: 'A node’s value is not its derivative. For a product, ∂y/∂a is the other input’s value.' },
        { t: 'num', q: 'y = a·b with a = 5 and b = −2. What is ∂y/∂a?', a: -2, tol: 0, ex: '∂y/∂a = b = −2: the other input’s value. a’s own value (5) is not its derivative.', gen: gen.valueTrap },
        { t: 'num', q: 'c = 4a and ∂L/∂c = 2.5. What gradient does a receive?', a: 10, tol: 0, ex: 'Local ∂c/∂a = 4. Incoming × local: 2.5 × 4 = 10.', gen: gen.local },
        { t: 'num', q: 'c = a·b with a = −3, b = 2, and ∂L/∂c = 4. What gradient does b receive?', a: -12, tol: 0, ex: 'Local ∂c/∂b = a = −3. Incoming × local: 4 × (−3) = −12.', gen: gen.local },
        { t: 'num', q: 'c = x² with x = −2, and ∂L/∂c = 3. What gradient is sent to x?', a: -12, tol: 0, ex: 'Local dc/dx = 2x = −4. Incoming × local: 3 × (−4) = −12.', gen: gen.local },
      ],
      cards: [
        { id: 'ad-incoming', front: 'The incoming gradient isn’t 1. What do you do?', back: 'Multiply it in: incoming × local.' },
        { id: 'ad-local-drill', front: 'c = a·b, a = 2, b = 3, ∂L/∂c = 5. Gradient to a?', back: '<b>15</b>.', gen: gen.cLocal },
        { id: 'ad-local-drill-2', front: 'c = x², x = 3, ∂L/∂c = 2. Gradient to x?', back: '<b>12</b>.', gen: gen.cLocal },
        { id: 'ad-local-rules', front: 'Local derivatives of a + b, a·b, x², k·a?', back: '1 and 1; b and a; 2x; k.' },
      ],
    },
    {
      id: 'ad-3',
      title: 'Paths, fast',
      steps: [
        { t: 'learn', title: 'Two rules for paths', body: '<div class="eq hook">Multiply along a path. Add across paths.</div><p>Write every node on the path; each edge contributes one factor. A variable can reach the output without being its parent.</p>' },
        { t: 'num', q: 'a = 2x, b = a + 1, y = b², at x = 1. What is dy/dx?', a: 12, tol: 0, ex: 'Values: a = 2, b = 3. Edges: dy/db = 2b = 6, db/da = 1, da/dx = 2. Multiply: 6 × 1 × 2 = 12.', gen: gen.chain3 },
        { t: 'mc', q: 'a = 2x, b = a + 1, y = b². Someone writes dy/dx = 2a · 2. What went wrong?', o: ['They skipped node b: dy/db = 2b, not 2a', 'Nothing', 'They missed a second path', 'They forgot the incoming gradient'], ex: 'y is the square of b, not of a. Skip an intermediate node and you use the wrong local derivative.' },
        { t: 'num', q: 'a = 2x, b = a + 5, y = 3b, at x = 4. x is not a parent of y. What is ∂y/∂x?', a: 6, tol: 0, ex: 'No direct edge, but a path: 2 × 1 × 3 = 6. Indirect influence still counts.', gen: gen.indirectOnly },
        { t: 'num', q: 'a = 2x, b = x², y = a + b, at x = 3. What is dy/dx?', a: 8, tol: 0, ex: 'Path through a: 2. Path through b: 2x = 6. Add across paths: 8.', gen: gen.branchSum },
        { t: 'num', q: 'x reaches y along two paths. One path’s product is 6, the other’s is −2. What is ∂y/∂x?', a: 4, tol: 0, ex: 'Accumulate means add: 6 + (−2) = 4.', gen: gen.accumulate },
        { t: 'num', q: 'a = 3x, b = 4x, y = a·b, at x = 2. What is dy/dx?', a: 48, tol: 0, ex: 'Values: a = 6, b = 8. Through a: b · 3 = 24. Through b: a · 4 = 24. Add: 48.', gen: gen.branchProduct },
        { t: 'num', q: 'a = 2x, b = x², y = a + b, at x = 3, with ∂L/∂y = 2. What is ∂L/∂x?', a: 16, tol: 0, ex: 'dy/dx = 2 + 6 = 8 (add across paths). Incoming isn’t 1: 2 × 8 = 16.', gen: gen.branchIncoming },
        { t: 'mc', q: '“Accumulate gradients” means…', o: ['Add each path’s contribution into the total', 'Keep the most recent contribution', 'Multiply the contributions', 'Average them'], ex: 'Every path carries part of the change, so the contributions add. Overwriting would drop a path.' },
      ],
      cards: [
        { id: 'ad-skip', front: 'Why not skip an intermediate node?', back: 'Each edge has its own local derivative; skipping a node uses the wrong one.' },
        { id: 'ad-branch', front: 'a = 2x, b = x², y = a + b, x = 3. dy/dx?', back: '2 + 6 = <b>8</b>.', gen: gen.cBranch },
        { id: 'ad-accumulate', front: '“Accumulate gradients” means?', back: 'Add the contributions.' },
        { id: 'ad-indirect', front: 'Can x affect y without being y’s parent?', back: 'Yes, through any path.' },
      ],
    },
    {
      id: 'ad-4',
      title: 'Full procedure',
      steps: [
        { t: 'learn', title: 'All five steps, no hints', body: '<p>Paths, values, one local derivative per edge, multiply along, add across. Then, if the incoming gradient isn’t 1, multiply by it.</p><div class="eq hook">Forward = values. Backward = sensitivities.</div>' },
        { t: 'num', q: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x. What is ∂y/∂x?', a: 67, tol: 0, ex: 'Indirect 1 × 4 × 8 × 2 = 64, direct 3: 67.', gen: gen.total },
        { t: 'num', q: 'x = 1, a = 3x, b = a², c = 2b, y = c + 5x. What is ∂y/∂x?', a: 41, tol: 0, ex: 'Values: a = 3, b = 9, c = 18. Indirect: 1 × 2 × 6 × 3 = 36. Direct: 5. Add: 41.', gen: gen.total },
        { t: 'num', q: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x, and y feeds the loss with ∂L/∂y = 2. What is ∂L/∂x?', a: 134, tol: 0, ex: '∂y/∂x = 64 + 3 = 67. The incoming gradient isn’t 1, so multiply it in: 2 × 67 = 134.', gen: gen.totalIncoming },
        { t: 'num', q: 'x = 1, a = x + 1, b = 3a, c = b², y = c + 2x. What is ∂y/∂x?', a: 38, tol: 0, ex: 'Values: a = 2, b = 6. Indirect: 1 × 12 × 3 × 1 = 36. Direct: 2. Add: 38.', gen: gen.shifted },
        { t: 'num', q: 'x = −1, a = 2x, b = a², c = 3b, y = c + 4x. What is ∂y/∂x?', a: -20, tol: 0, ex: 'Values: a = −2, b = 4. Indirect: 1 × 3 × (−4) × 2 = −24. Direct: 4. Add: −20.', gen: gen.total },
        { t: 'num', q: 'x = 3, a = 2x, b = a², c = 2b, y = c − x, and ∂L/∂y = −1. What is ∂L/∂x?', a: -47, tol: 0, ex: '∂y/∂x = 1 × 2 × 12 × 2 + (−1) = 47. Incoming −1: −47.', gen: gen.totalIncoming },
        { t: 'mc', q: 'A network has one scalar loss and millions of parameters. Which mode gives every gradient in one backward pass?', o: ['Reverse mode', 'Forward mode', 'Either: they cost the same', 'Neither'], ex: 'One scalar loss, many parameters → reverse mode: one backward pass from ∂L/∂L = 1 reaches every parameter.' },
      ],
      cards: [
        { id: 'ad-hooks', front: 'The four memory hooks?', back: 'Incoming × local. Multiply along paths; add across. Forward = values, backward = sensitivities. One scalar loss, many parameters → reverse mode.' },
        { id: 'ad-full', front: 'x = 1, a = 3x, b = a², c = 2b, y = c + 5x. ∂y/∂x?', back: '1 × 2 × 6 × 3 + 5 = <b>41</b>.', gen: gen.cTotal },
        { id: 'ad-full-incoming', front: 'x = 2, a = 2x, b = a², c = 4b, y = c + 3x, ∂L/∂y = 2. ∂L/∂x?', back: '2 × 67 = <b>134</b>.', gen: gen.cIncoming },
        { id: 'ad-reverse', front: 'One scalar loss, many parameters → ?', back: 'Reverse mode.' },
      ],
    },
  ],
};
