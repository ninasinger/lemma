// Unit: derivatives, the chain rule and backprop on scalar graphs, then the vector version.
// Scalar rules come first and are drilled until automatic; Jacobians only appear in the last lesson.
// Notation: d for one input, ∂ when a node has several inputs. ∂L/∂(node) is that node's gradient.
import { num, term, list, mtx, big, py, dot, dotTerms } from '../vary.js';
import { SUB } from '../util.js';

// Fresh numbers every time a question or card is shown (see ../vary.js). Each py recomputes the
// answer with deriv(), a complex-step derivative defined in scripts/verify_numpy.py.
const plus = (k) => (k < 0 ? `− ${-k}` : `+ ${k}`);
const col = (M, j) => M.map((row) => row[j]);
const other = { a: 'b', b: 'a' };
const gen = {
  nudgeLine(v) {
    const c = v.int(2, 9), x = v.int(1, 9), dx = v.pick([0.01, 0.1, 0.001]), a = +(c * dx).toFixed(6);
    return { q: `y = ${num(c)}x. If x goes from ${num(x)} to ${num(x + dx)}, by how much does y change?`, a,
      ex: `dy/dx = ${num(c)} everywhere, so Δy = ${num(c)} × ${num(dx)} = ${num(a)}. For a straight line the slope gives the change exactly.`,
      py: `f = lambda x: ${c} * x\nans = f(${x} + ${dx}) - f(${x})` };
  },
  slopeSquare(v) {
    const x = v.nz(-6, 9);
    return { q: `y = x². What is dy/dx at x = ${num(x)}?`, a: 2 * x, ex: `d(x²)/dx = 2x, and 2 × ${term(x)} = ${num(2 * x)}.`, py: `ans = deriv(lambda x: x**2, ${x})` };
  },
  slopeLine(v) {
    const c = v.int(2, 9), k = v.nz(-9, 9), x = v.int(2, 20);
    return { q: `y = ${num(c)}x ${plus(k)}. What is dy/dx at x = ${num(x)}?`, a: c,
      ex: `The slope of ${num(c)}x ${plus(k)} is ${num(c)} everywhere: x = ${num(x)} doesn’t change it, and the constant contributes 0.`, py: `ans = deriv(lambda x: ${c} * x + ${k}, ${x})` };
  },
  valueVsSlope(v) {
    const x = v.int(3, 9);
    return { q: `At x = ${num(x)}, y = x² has the value ${num(x * x)}. What is its derivative there?`, o: [`${2 * x}`, `${x * x}`, `${x}`, '2'],
      ex: `Value and slope are different numbers. y = ${num(x * x)} says where you are; dy/dx = 2 × ${num(x)} = ${num(2 * x)} says how fast y changes there.` };
  },
  nudgeSquare(v) {
    const x = v.int(1, 9), a = +(2 * x * 0.01).toFixed(6);
    return { q: `y = x². If x goes from ${num(x)} to ${num(x + 0.01)}, about how much does y change?`, a,
      ex: `dy/dx = 2 × ${num(x)} = ${num(2 * x)}, so Δy ≈ ${num(2 * x)} × 0.01 = ${num(a)}. (Exactly: ${num(x + 0.01)}² − ${num(x * x)} = ${num((x + 0.01) ** 2 - x * x)}.)`,
      py: `ans = (${x} + 0.01)**2 - ${x}**2` };
  },

  mulLocal(v) {
    const a = v.nz(-5, 9), b = v.nz(-5, 9), w = v.pick(['a', 'b']), ov = w === 'a' ? b : a;
    return { q: `y = a·b with a = ${num(a)} and b = ${num(b)}. What is ∂y/∂${w}?`, a: ov,
      ex: `Hold ${other[w]} fixed: y = ${num(ov)}·${w}, a line with slope ${num(ov)}. So ∂y/∂${w} = ${other[w]} = ${num(ov)}: the other input’s value.`,
      py: `ans = deriv(lambda a, b: a * b, ${a}, ${b}, wrt=${w === 'a' ? 0 : 1})` };
  },
  addLocal(v) {
    const a = v.nz(-9, 9), b = v.nz(-9, 9), w = v.pick(['a', 'b']);
    return { q: `y = a + b with a = ${num(a)} and b = ${num(b)}. What is ∂y/∂${w}?`, a: 1,
      ex: `Nudge ${w} by 0.01 and y moves by exactly 0.01. ∂y/∂${w} = 1, whatever the values.`, py: `ans = deriv(lambda a, b: a + b, ${a}, ${b}, wrt=${w === 'a' ? 0 : 1})` };
  },
  mulPlusC(v) {
    const a = v.nz(-5, 6), b = v.nz(-5, 6), c = v.int(2, 12), w = v.pick(['a', 'b', 'c']);
    const ans = w === 'a' ? b : w === 'b' ? a : 1;
    const ex = w === 'c' ? 'c is added directly, so ∂y/∂c = 1.' : `The + c doesn’t change the slope. Holding ${other[w]} fixed, a·b changes at rate ${other[w]} = ${num(ans)}.`;
    return { q: `y = a·b + c with a = ${num(a)}, b = ${num(b)} and c = ${num(c)}. What is ∂y/∂${w}?`, a: ans, ex,
      py: `ans = deriv(lambda a, b, c: a * b + c, ${a}, ${b}, ${c}, wrt=${'abc'.indexOf(w)})` };
  },

  link1(v) {
    const c = v.int(2, 5), x = v.nz(-4, 5);
    return { q: `a = ${num(c)}x and y = a². At x = ${num(x)}, what is da/dx?`, a: c, ex: `a = ${num(c)}x is a line with slope ${num(c)}: da/dx = ${num(c)} at any x.`, py: `ans = deriv(lambda x: ${c} * x, ${x})` };
  },
  link2(v) {
    const c = v.int(2, 5), x = v.nz(-4, 5), a = c * x;
    return { q: `a = ${num(c)}x and y = a². At x = ${num(x)}, so a = ${num(a)}, what is dy/da?`, a: 2 * a, ex: `dy/da = 2a = 2 × ${term(a)} = ${num(2 * a)}.`, py: `ans = deriv(lambda a: a**2, ${a})` };
  },
  chain2(v) {
    const c = v.int(2, 5), x = v.nz(-4, 5), a = c * x;
    return { q: `a = ${num(c)}x and y = a². At x = ${num(x)}, what is dy/dx?`, a: 2 * a * c,
      ex: `Forward: a = ${num(a)}. Local derivatives: da/dx = ${num(c)} and dy/da = 2a = ${num(2 * a)}. Multiply along the path: ${term(2 * a)} × ${num(c)} = ${num(2 * a * c)}.`,
      py: `ans = deriv(lambda x: (${c} * x)**2, ${x})` };
  },
  chainShift(v) {
    const k = v.nz(-9, 9), c = v.nz(-5, 7), x = v.int(1, 20);
    return { q: `a = x ${plus(k)} and y = ${num(c)}a. What is dy/dx at x = ${num(x)}?`, a: c,
      ex: `da/dx = 1 and dy/da = ${num(c)}. Multiply: ${term(c)} × 1 = ${num(c)}. In a chain of straight lines the values don’t matter.`, py: `ans = deriv(lambda x: ${c} * (x + ${k}), ${x})` };
  },
  chain3(v) {
    const c = v.int(2, 4), k = v.nz(-3, 3), x = v.nz(-2, 3), a = c * x, b = a + k;
    return { q: `Three links: a = ${num(c)}x, b = a ${plus(k)}, y = b². At x = ${num(x)}, what is dy/dx?`, a: 2 * b * c,
      ex: `Forward: a = ${num(a)}, b = ${num(b)}. Local: da/dx = ${num(c)}, db/da = 1, dy/db = 2b = ${num(2 * b)}. Multiply along the path: ${term(2 * b)} × 1 × ${num(c)} = ${num(2 * b * c)}.`,
      py: `ans = deriv(lambda x: (${c} * x + ${k})**2, ${x})` };
  },
  chainSymbolic(v) {
    const c = v.int(2, 5);
    return { q: `a = ${num(c)}x and y = a². Which is dy/dx?`, o: [`${2 * c * c}x`, `${2 * c}x`, '2a', `${c * c}x²`],
      ex: `dy/da · da/dx = 2a · ${num(c)} = ${num(2 * c)}a, and a = ${num(c)}x, so ${num(2 * c * c)}x. Stopping at 2a skips the da/dx link; ${num(c * c)}x² is y itself.` };
  },

  indirect(v) {
    const c = v.int(2, 5), k = v.nz(-6, 6), m = v.nz(-4, 5), x = v.int(1, 9);
    return { q: `x → a → b → y with a = ${num(c)}x, b = a ${plus(k)} and y = ${num(m)}b. x is not a parent of y. What is ∂y/∂x at x = ${num(x)}?`, a: c * m,
      ex: `Not a parent, but there is a path: da/dx · db/da · dy/db = ${num(c)} × 1 × ${term(m)} = ${num(c * m)}. Influence travels through a and b.`,
      py: `ans = deriv(lambda x: ${m} * (${c} * x + ${k}), ${x})` };
  },

  pathA(v) {
    const c = v.int(2, 6), x = v.nz(-4, 6);
    return { q: `a = ${num(c)}x, b = x² and y = a + b. At x = ${num(x)}, what does the path through a contribute to dy/dx?`, a: c,
      ex: `Along x → a → y: da/dx = ${num(c)} and dy/da = 1. Multiply: ${num(c)} × 1 = ${num(c)}.`, py: `xb = ${x}\nans = deriv(lambda x: ${c} * x + xb**2, ${x})` };
  },
  pathB(v) {
    const c = v.int(2, 6), x = v.nz(-4, 6);
    return { q: `a = ${num(c)}x, b = x² and y = a + b. At x = ${num(x)}, what does the path through b contribute to dy/dx?`, a: 2 * x,
      ex: `Along x → b → y: db/dx = 2x = ${num(2 * x)} and dy/db = 1. Multiply: ${num(2 * x)}.`, py: `xa = ${x}\nans = deriv(lambda x: ${c} * xa + x**2, ${x})` };
  },
  branchTotal(v) {
    const c = v.int(2, 6), x = v.nz(-4, 6);
    return { q: `a = ${num(c)}x, b = x² and y = a + b. At x = ${num(x)}, what is the total dy/dx?`, a: c + 2 * x,
      ex: `Through a: ${num(c)}. Through b: 2x = ${num(2 * x)}. Add across paths: ${num(c)} + ${term(2 * x)} = ${num(c + 2 * x)}.`, py: `ans = deriv(lambda x: ${c} * x + x**2, ${x})` };
  },
  prodBranch(v) {
    const p = v.int(2, 5), q = v.int(2, 5), x = v.nz(-3, 4), a = p * x, b = q * x;
    return { q: `a = ${num(p)}x, b = ${num(q)}x and y = a·b. At x = ${num(x)}, what is dy/dx?`, a: b * p + a * q,
      ex: `Through a: dy/da · da/dx = b · ${num(p)} = ${term(b)} × ${num(p)} = ${num(b * p)}. Through b: a · ${num(q)} = ${term(a)} × ${num(q)} = ${num(a * q)}. Add across paths: ${num(b * p + a * q)}.`,
      py: `ans = deriv(lambda x: (${p} * x) * (${q} * x), ${x})` };
  },

  mulToA(v) {
    const a = v.nz(-4, 6), b = v.nz(-4, 6), g = v.nz(-3, 7);
    return { q: `c = a·b with a = ${num(a)} and b = ${num(b)}, and the incoming gradient is ∂L/∂c = ${num(g)}. What gradient does a receive?`, a: g * b,
      ex: `Local derivative ∂c/∂a = b = ${num(b)}. Incoming × local: ${term(g)} × ${term(b)} = ${num(g * b)}.`, py: `ans = ${g} * deriv(lambda a, b: a * b, ${a}, ${b}, wrt=0)` };
  },
  mulToB(v) {
    const a = v.nz(-4, 6), b = v.nz(-4, 6), g = v.nz(-3, 7);
    return { q: `c = a·b with a = ${num(a)} and b = ${num(b)}, and ∂L/∂c = ${num(g)}. What gradient does b receive?`, a: g * a,
      ex: `Local derivative ∂c/∂b = a = ${num(a)}. Incoming × local: ${term(g)} × ${term(a)} = ${num(g * a)}.`, py: `ans = ${g} * deriv(lambda a, b: a * b, ${a}, ${b}, wrt=1)` };
  },
  sumTo(v) {
    const a = v.nz(-9, 9), b = v.nz(-9, 9), g = v.nz(-6, 9), w = v.pick(['a', 'b']);
    return { q: `c = a + b with a = ${num(a)} and b = ${num(b)}, and ∂L/∂c = ${num(g)}. What gradient does ${w} receive?`, a: g,
      ex: `∂c/∂${w} = 1, so ${w} gets ${term(g)} × 1 = ${num(g)}. A sum hands the incoming gradient to each input unchanged.`,
      py: `ans = ${g} * deriv(lambda a, b: a + b, ${a}, ${b}, wrt=${w === 'a' ? 0 : 1})` };
  },
  passBack(v) {
    const g = v.nz(-4, 9), l = v.nz(-4, 9);
    return { q: `The incoming gradient is ${num(g)} and the local derivative is ${num(l)}. What gets passed backward?`, a: g * l,
      ex: `Incoming × local: ${term(g)} × ${term(l)} = ${num(g * l)}. Not ${num(g)} (that ignores the local rule) and not ${num(l)} (that ignores everything downstream).`,
      py: `ans = deriv(lambda u: ${g} * u, 0.0) * deriv(lambda x: ${l} * x, 0.0)` };
  },
  squareTo(v) {
    const x = v.nz(-4, 5), g = v.nz(-3, 5);
    return { q: `c = x² with x = ${num(x)}, and ∂L/∂c = ${num(g)}. What gradient is sent to x?`, a: g * 2 * x,
      ex: `Local: dc/dx = 2x = ${num(2 * x)}. Incoming × local: ${term(g)} × ${term(2 * x)} = ${num(g * 2 * x)}.`, py: `ans = ${g} * deriv(lambda x: x**2, ${x})` };
  },
  scaleTo(v) {
    const k = v.pick([2, 3, 4, -3, 10]), g = v.pick([0.5, 1.5, 2.5, -2, 4]), a = +(g * k).toFixed(6);
    return { q: `c = ${num(k)}a and ∂L/∂c = ${num(g)}. What gradient does a receive?`, a,
      ex: `Local: ∂c/∂a = ${num(k)}. Incoming × local: ${num(g)} × ${term(k)} = ${num(a)}.`, py: `ans = ${g} * deriv(lambda a: ${k} * a, 1.0)` };
  },

  lossA(v) {
    const x = v.int(1, 4), w = v.nz(-3, 4), y = v.int(-3, 6), a = x * w;
    return { q: `a = x·w and L = (a − y)², with x = ${num(x)}, w = ${num(w)} and y = ${num(y)}. Forward: a = ${num(a)}. Starting from ∂L/∂L = 1, what is ∂L/∂a?`, a: 2 * (a - y),
      ex: `The local derivative of (a − y)² with respect to a is 2(a − y) = 2 × (${num(a)} − ${term(y)}) = ${num(2 * (a - y))}. Times the incoming 1: ${num(2 * (a - y))}.`,
      py: `ans = 1 * deriv(lambda a: (a - ${y})**2, ${a})` };
  },
  lossW(v) {
    const x = v.int(1, 4), w = v.nz(-3, 4), y = v.int(-3, 6), a = x * w, ga = 2 * (a - y);
    return { q: `a = x·w and L = (a − y)², with x = ${num(x)}, w = ${num(w)} and y = ${num(y)}. What is ∂L/∂w?`, a: ga * x,
      ex: `Forward: a = ${num(a)}. Backward: ∂L/∂a = 2(a − y) = ${num(ga)}. Then w gets incoming × local = ${term(ga)} × x = ${term(ga)} × ${num(x)} = ${num(ga * x)}.`,
      py: `ans = deriv(lambda w: (${x} * w - ${y})**2, ${w})` };
  },
  passes(v) {
    const [nIn, nOut] = v.pick([[1000000, 1], [3, 1000], [50000, 1], [2, 10000], [784, 10]]), mode = v.pick(['forward', 'reverse']);
    return { q: `f maps ${big(nIn)} input${nIn > 1 ? 's' : ''} to ${big(nOut)} output${nOut > 1 ? 's' : ''}. How many ${mode}-mode passes give every derivative of every output?`,
      a: mode === 'forward' ? nIn : nOut,
      ex: mode === 'forward' ? `Forward mode carries the derivatives with respect to one input per pass: ${big(nIn)} passes. Reverse mode would take ${big(nOut)}.` : `Reverse mode gives the derivatives of one output with respect to every input per pass: ${big(nOut)} pass${nOut > 1 ? 'es' : ''}. Forward mode would take ${big(nIn)}.`,
      py: `ans = ${mode === 'forward' ? nIn : nOut}` };
  },

  engineXX(v) {
    const x = v.nz(-5, 9);
    return { q: `Engine trace: a = x·x with x = ${num(x)} (x feeds both inputs) and L = a. After L.backward(), what is x.grad?`, a: 2 * x,
      ex: `The product’s backward rule runs once per input slot: x.grad += 1 × ${term(x)}, twice. Accumulate: ${num(x)} + ${term(x)} = ${num(2 * x)}, which is d(x²)/dx = 2x.`,
      py: `ans = deriv(lambda x: x * x, ${x})` };
  },
  engineOverwrite(v) {
    const x = v.nz(-5, 9);
    return { q: `Same graph, a = x·x with x = ${num(x)} and L = a, but the backward rule overwrites: <code>x.grad = out.grad * other.data</code>. What does x.grad end up as?`, a: x,
      ex: `Both writes set ${num(x)}; the second replaces the first, leaving ${num(x)}. The true dL/dx is 2x = ${num(2 * x)}: overwriting lost a path.`,
      py: `g = 0.0\nfor _ in range(2):\n    g = 1.0 * ${x}\nans = g` };
  },
  engineMix(v) {
    const a = v.nz(-4, 6), b = v.nz(-4, 6);
    return { q: `c = a·b and L = c + a, with a = ${num(a)} and b = ${num(b)}. After L.backward(), what is a.grad?`, a: b + 1,
      ex: `Two paths into a. Through c: 1 × b = ${num(b)}. Directly into L: 1. Accumulate: ${num(b)} + 1 = ${num(b + 1)}.`, py: `ans = deriv(lambda a: a * ${b} + a, ${a})` };
  },

  jacSize(v) {
    const n = v.pick([3, 4, 10, 100, 784]), m = v.pick([2, 3, 5, 10, 64]);
    return { q: `f maps ${n} inputs to ${num(m)} outputs. How many entries does its Jacobian have?`, a: m * n,
      ex: `J has shape (outputs, inputs) = (${num(m)}, ${n}): ${big(m * n)} entries, one ∂yᵢ/∂xⱼ for each pair.`, py: `ans = np.zeros((${m}, ${n})).size` };
  },
  jacEntry(v) {
    const W = v.mat(2, 3, -4, 6), i = v.int(0, 1), j = v.int(0, 2);
    return { q: `y = W x with W = ${mtx(W)}. What is ∂y${SUB[i]}/∂x${SUB[j]}?`, a: W[i][j],
      ex: `y${SUB[i]} = ${W[i].map((w, k) => `${term(w)}·x${SUB[k]}`).join(' + ')}, so ∂y${SUB[i]}/∂x${SUB[j]} = W[${i}, ${j}] = ${num(W[i][j])}. The Jacobian of Wx is W itself.`,
      py: `W = np.array(${py(W)})\nans = deriv(lambda *x: (W @ np.array(x))[${i}], 1.0, 1.0, 1.0, wrt=${j})` };
  },
  vjp(v) {
    const W = v.mat(2, 2, -3, 4), g = [v.nz(-3, 3), v.nz(-3, 3)], j = v.int(0, 1), s = dot(col(W, j), g);
    return { q: `y = W x with W = ${mtx(W)} and incoming gradient g = ∂L/∂y = ${list(g)}. What is entry ${j} of ∂L/∂x = Wᵀg?`, a: s,
      ex: `Entry ${j} is column ${j} of W dotted with g: ${dotTerms(col(W, j), g)} = ${num(s)}. Each x${SUB[j]} collects the gradient from every output it feeds.`,
      py: `W, g = np.array(${py(W)}), np.array(${py(g)})\nans = deriv(lambda *x: g @ (W @ np.array(x)), 0.0, 0.0, wrt=${j})` };
  },
  elemJac(v) {
    const n = v.pick([100, 1000, 4096, 10000]);
    return { q: `An elementwise operation (like ReLU) maps ${big(n)} values to ${big(n)} values. How many entries does its full Jacobian have?`, a: n * n,
      ex: `${big(n)} × ${big(n)} = ${big(n * n)}. Only the ${big(n)} diagonal entries can be non-zero, so backprop never builds it: it multiplies g elementwise by f′(x).`,
      py: `ans = np.zeros(${n}).size ** 2` };
  },

  // Flashcards: same idea, new numbers.
  cSquare(v) {
    const x = v.nz(-6, 9);
    return { front: `d/dx of x² at x = ${num(x)}?`, back: `2x = <b>${num(2 * x)}</b>.`, a: 2 * x, py: `ans = deriv(lambda x: x**2, ${x})` };
  },
  cNudge(v) {
    const d = v.nz(-5, 9), dx = v.pick([0.01, 0.1]);
    return { front: `dy/dx = ${num(d)}. x moves by ${num(dx)}. Δy ≈ ?`, back: `${num(d)} × ${num(dx)} = <b>${num(d * dx)}</b>.`, a: +(d * dx).toFixed(6), py: `ans = ${d} * ${dx}` };
  },
  cMul(v) {
    const a = v.nz(-4, 9), b = v.nz(-4, 9);
    return { front: `y = a·b, a = ${num(a)}, b = ${num(b)}. ∂y/∂a and ∂y/∂b?`, back: `<b>${num(b)}</b> and <b>${num(a)}</b>: each is the other input’s value.`, a: [b, a],
      py: `ans = [deriv(lambda a, b: a * b, ${a}, ${b}, wrt=i) for i in (0, 1)]` };
  },
  cChain(v) {
    const c = v.int(2, 4), x = v.nz(-3, 4), a = c * x;
    return { front: `a = ${num(c)}x, y = a², x = ${num(x)}. dy/dx?`, back: `2a · ${num(c)} = ${term(2 * a)} × ${num(c)} = <b>${num(2 * a * c)}</b>.`, a: 2 * a * c, py: `ans = deriv(lambda x: (${c} * x)**2, ${x})` };
  },
  cChainSym(v) {
    const c = v.int(2, 5);
    return { front: `a = ${num(c)}x, y = a². dy/dx in terms of x?`, back: `2a · ${num(c)} = <b>${num(2 * c * c)}x</b>.`, a: 2 * c * c, py: `ans = deriv(lambda x: (${c} * x)**2, 1.0)` };
  },
  cBranch(v) {
    const c = v.int(2, 6), x = v.nz(-4, 6);
    return { front: `a = ${num(c)}x, b = x², y = a + b, x = ${num(x)}. dy/dx?`, back: `${num(c)} + ${term(2 * x)} = <b>${num(c + 2 * x)}</b>.`, a: c + 2 * x, py: `ans = deriv(lambda x: ${c} * x + x**2, ${x})` };
  },
  cSend(v) {
    const g = v.nz(-4, 9), l = v.nz(-4, 9);
    return { front: `Incoming ${num(g)}, local ${num(l)}. Passed back?`, back: `<b>${num(g * l)}</b>.`, a: g * l, py: `ans = ${g} * ${l}` };
  },
  cMulBack(v) {
    const a = v.nz(-4, 6), b = v.nz(-4, 6), g = v.nz(-3, 7);
    return { front: `c = a·b, a = ${num(a)}, b = ${num(b)}, ∂L/∂c = ${num(g)}. Gradients to a and b?`, back: `a: ${term(g)} × ${term(b)} = <b>${num(g * b)}</b>. b: ${term(g)} × ${term(a)} = <b>${num(g * a)}</b>.`, a: [g * b, g * a],
      py: `ans = [${g} * deriv(lambda a, b: a * b, ${a}, ${b}, wrt=i) for i in (0, 1)]` };
  },
  cAddBack(v) {
    const g = v.nz(-6, 9);
    return { front: `c = a + b, ∂L/∂c = ${num(g)}. Gradients to a and b?`, back: `<b>${num(g)}</b> and <b>${num(g)}</b>: a sum passes the gradient through unchanged.`, a: [g, g],
      py: `ans = [${g} * deriv(lambda a, b: a + b, 1.0, 2.0, wrt=i) for i in (0, 1)]` };
  },
  cSqLoss(v) {
    const a = v.int(-3, 8), y = v.int(-3, 6);
    return { front: `L = (a − y)², a = ${num(a)}, y = ${num(y)}. ∂L/∂a?`, back: `2(a − y) = <b>${num(2 * (a - y))}</b>.`, a: 2 * (a - y), py: `ans = deriv(lambda a: (a - ${y})**2, ${a})` };
  },
  cEngineXX(v) {
    const x = v.nz(-5, 9);
    return { front: `a = x·x, x = ${num(x)}, L = a. x.grad after backward?`, back: `<b>${num(2 * x)}</b>: ${num(x)} + ${term(x)}, one per input slot.`, a: 2 * x, py: `ans = deriv(lambda x: x * x, ${x})` };
  },
  cJac(v) {
    const n = v.pick([3, 10, 784]), m = v.pick([2, 5, 64]);
    return { front: `Jacobian of f: ${n} inputs → ${num(m)} outputs. Shape?`, back: `<b>(${num(m)}, ${n})</b>: J[i, j] = ∂yᵢ/∂xⱼ, outputs by inputs.`, a: `(${m}, ${n})`, py: `ans = np.zeros((${m}, ${n})).shape` };
  },
};

export default {
  id: 'dv',
  title: 'Derivatives',
  sub: 'Local slopes, the chain rule, and how gradients flow backward through a graph.',
  c: 'dv',
  guide: `<h3>Three things to remember</h3><div class="eq hook">gradient to parent = incoming gradient × local derivative</div><div class="eq hook">Multiply along paths; add across paths.</div><div class="eq hook">Forward computes values. Backward computes sensitivities.</div>
  <h3>Local derivatives</h3><ul><li>Constant: 0. x: 1. cx: c. x²: 2x. Differentiate first, then plug in the value.</li><li>y = a + b: ∂y/∂a = 1, ∂y/∂b = 1. The values don’t matter.</li><li>y = a·b: ∂y/∂a = b, ∂y/∂b = a. Each is the <b>other</b> input’s value.</li><li>A node’s value and its local derivative are different numbers.</li></ul>
  <h3>Chain rule</h3><ul><li>One path: multiply the local derivatives. a = 2x, y = a² at x = 3: 2a · 2 = 12 · 2 = 24.</li><li>Several paths: compute each path’s product, then add. a = 2x, b = x², y = a + b at x = 3: 2 + 6 = 8.</li><li>Accumulate = add. A variable that isn’t a direct parent can still have a non-zero derivative.</li></ul>
  <h3>Backprop</h3><ul><li>Forward pass: compute values and record the graph.</li><li>Backward pass: start at the scalar loss with ∂L/∂L = 1, visit nodes in reverse topological order, send incoming × local to each parent, and accumulate with +=.</li><li>Each node stores its value, grad, parents and backward rule. Zero the grads before each backward pass.</li><li>Reverse mode: one scalar loss, many parameters, one backward pass. Forward mode suits few inputs, many outputs.</li></ul>
  <h3>Vectors</h3><ul><li>The Jacobian of f: ℝⁿ → ℝᵐ has shape (m, n), J[i, j] = ∂yᵢ/∂xⱼ. For y = Wx, J = W.</li><li>Backprop sends back gᵀJ (a vector-Jacobian product), never J itself. For y = Wx: ∂L/∂x = Wᵀg.</li></ul>`,
  lessons: [
    {
      id: 'dv-1',
      title: 'What a derivative means',
      steps: [
        { t: 'learn', title: 'A local exchange rate', body: '<p>The derivative <span class="m">dy/dx</span> answers one question: <b>if x changes a little, how much does y change?</b> It is the slope of y at that point.</p><p>If dy/dx = 3, nudging x by 0.01 moves y by about 0.03.</p>' },
        { t: 'num', q: 'y = 4x. If x goes from 2 to 2.01, by how much does y change?', a: 0.04, tol: 0.0005, ex: 'dy/dx = 4 everywhere, so Δy = 4 × 0.01 = 0.04.', gen: gen.nudgeLine },
        { t: 'learn', title: 'Four rules to start', body: '<div class="eq hook">constant → 0 · x → 1 · cx → c · x² → 2x</div><p>Differentiate first, then plug in the value: d(x²)/dx = 2x, so at x = 3 the slope is 6.</p>' },
        { t: 'match', q: 'Match each function to its derivative.', p: [['y = 7', '0'], ['y = x', '1'], ['y = 5x', '5'], ['y = x²', '2x']], ex: 'A constant never moves; x moves one-for-one; cx moves c-for-one; x² moves faster the bigger x is.' },
        { t: 'num', q: 'y = x². What is dy/dx at x = 3?', a: 6, tol: 0, ex: 'd(x²)/dx = 2x, and 2 × 3 = 6.', gen: gen.slopeSquare },
        { t: 'num', q: 'y = 5x + 2. What is dy/dx at x = 10?', a: 5, tol: 0, ex: 'The slope of 5x + 2 is 5 everywhere: x = 10 doesn’t change it, and the constant contributes 0.', gen: gen.slopeLine },
        { t: 'mc', q: 'At x = 3, y = x² has the value 9. What is its derivative there?', o: ['6', '9', '3', '2'], ex: 'Value and slope are different numbers. y = 9 says where you are; dy/dx = 2 × 3 = 6 says how fast y changes there.', gen: gen.valueVsSlope },
        { t: 'num', q: 'y = x². If x goes from 3 to 3.01, about how much does y change?', a: 0.06, tol: 0.002, ex: 'dy/dx = 6, so Δy ≈ 6 × 0.01 = 0.06. (Exactly: 3.01² − 9 = 0.0601.)', gen: gen.nudgeSquare },
      ],
      cards: [
        { id: 'dv-mean', front: 'What does dy/dx measure?', back: 'How much y changes per small change in x: the local slope.' },
        { id: 'dv-const', front: 'd/dx of a constant?', back: '0.' },
        { id: 'dv-cx', front: 'd/dx of cx? Of x?', back: 'c. And 1.' },
        { id: 'dv-x2', front: 'd/dx of x² at x = 3?', back: '2x = <b>6</b>.', gen: gen.cSquare },
        { id: 'dv-nudge', front: 'dy/dx = 4. x moves by 0.01. Δy ≈ ?', back: '<b>0.04</b>.', gen: gen.cNudge },
      ],
    },
    {
      id: 'dv-2',
      title: 'Local derivatives',
      steps: [
        { t: 'learn', title: 'One operation, one rule', body: '<p>A computation is built from small operations. Each has a <b>local derivative</b> for each input: how its output moves when that input moves and the others stay put.</p><div class="eq">a + b: ∂/∂a = 1, ∂/∂b = 1</div><div class="eq">a·b: ∂/∂a = b, ∂/∂b = a</div><div class="eq">x²: d/dx = 2x</div>' },
        { t: 'mc', q: 'y = b + x. What is ∂y/∂x?', o: ['1', 'b', 'x', '0'], ex: 'Adding b just shifts y. Nudge x by 0.01 and y moves by exactly 0.01, whatever b is. The answer is 1, not b.' },
        { t: 'num', q: 'y = a·b with a = 4 and b = 7. What is ∂y/∂a?', a: 7, tol: 0, ex: 'Hold b fixed: y = 7a, a line with slope 7. So ∂y/∂a = b = 7: the other input’s value.', gen: gen.mulLocal },
        { t: 'mc', q: 'd = c + a. What is ∂d/∂c?', o: ['1', '0', 'a', 'c'], ex: 'c is a direct input to a sum, so ∂d/∂c = 1. It would be 0 only if d didn’t depend on c at all.' },
        { t: 'learn', title: 'Value vs local derivative', body: '<p>Each node has a <b>value</b> from the forward pass and, for each input, a <b>local derivative</b>.</p><p>y = a·b with a = 2, b = 3: the value is y = 6. ∂y/∂a = 3 and ∂y/∂b = 2. Three different numbers with three different jobs.</p>' },
        { t: 'mc', q: 'y = a·b with a = 2 and b = 3. Is the value of a (2) the same as ∂y/∂a?', o: ['No: ∂y/∂a = b = 3', 'Yes: both are 2', 'No: ∂y/∂a = 6', 'No: ∂y/∂a = 1'], ex: 'A parent’s value is not its local derivative. For a product, ∂y/∂a is the <b>other</b> input’s value: 3.' },
        { t: 'num', q: 'y = a + b with a = 9 and b = −4. What is ∂y/∂b?', a: 1, tol: 0, ex: 'Nudge b by 0.01 and y moves by exactly 0.01. ∂y/∂b = 1, whatever the values.', gen: gen.addLocal },
        { t: 'match', q: 'Match each local derivative to its value.', p: [['∂(a + b)/∂a', '1'], ['∂(a·b)/∂a', 'b'], ['∂(a·b)/∂b', 'a'], ['d(x²)/dx', '2x']], ex: 'Sums pass changes straight through; products scale by the other input; squares by twice the input.' },
        { t: 'num', q: 'y = a·b + c with a = 3, b = −2 and c = 10. What is ∂y/∂b?', a: 3, tol: 0, ex: 'The + c doesn’t change the slope. Holding a fixed, a·b changes at rate a = 3.', gen: gen.mulPlusC },
      ],
      cards: [
        { id: 'dv-add', front: '∂(a + b)/∂a?', back: '1. Same for b.' },
        { id: 'dv-mul', front: '∂(a·b)/∂a?', back: 'b, the other input.' },
        { id: 'dv-trap-sum', front: 'y = b + x. ∂y/∂x?', back: '1, not b.' },
        { id: 'dv-value-local', front: 'Is a parent’s value the same as its local derivative?', back: 'No. For y = a·b, ∂y/∂a is b’s value.' },
        { id: 'dv-mul-num', front: 'y = a·b, a = 4, b = 7. ∂y/∂a and ∂y/∂b?', back: '<b>7</b> and <b>4</b>.', gen: gen.cMul },
      ],
    },
    {
      id: 'dv-3',
      title: 'Chain rule on one path',
      steps: [
        { t: 'learn', title: 'Two links', body: '<p>a = 2x and y = a². To get dy/dx: compute the values, find each local derivative, then <b>multiply</b>.</p><div class="eq">dy/dx = dy/da · da/dx</div><p>At x = 3: a = 6. da/dx = 2 and dy/da = 2a = 12, so dy/dx = 12 × 2 = 24.</p><p><b>Multiply along a path.</b></p>' },
        { t: 'num', q: 'a = 2x and y = a². At x = 3, what is da/dx?', a: 2, tol: 0, ex: 'a = 2x is a line with slope 2: da/dx = 2 at any x.', gen: gen.link1 },
        { t: 'num', q: 'a = 2x and y = a². At x = 3, so a = 6, what is dy/da?', a: 12, tol: 0, ex: 'dy/da = 2a = 2 × 6 = 12.', gen: gen.link2 },
        { t: 'num', q: 'a = 2x and y = a². At x = 3, what is dy/dx?', a: 24, tol: 0, ex: 'Forward: a = 6. Local derivatives: da/dx = 2 and dy/da = 12. Multiply along the path: 12 × 2 = 24.', gen: gen.chain2 },
        { t: 'mc', q: 'What do you do with the local derivatives along one path?', o: ['Multiply them', 'Add them', 'Keep the largest', 'Average them'], ex: 'Each link scales the change from the link before it, so the scale factors multiply.' },
        { t: 'num', q: 'a = x + 4 and y = 3a. What is dy/dx at x = 10?', a: 3, tol: 0, ex: 'da/dx = 1 and dy/da = 3. Multiply: 3 × 1 = 3. In a chain of straight lines the values don’t matter.', gen: gen.chainShift },
        { t: 'num', q: 'Three links: a = 3x, b = a + 1, y = b². At x = 1, what is dy/dx?', a: 24, tol: 0, ex: 'Forward: a = 3, b = 4. Local: da/dx = 3, db/da = 1, dy/db = 2b = 8. Multiply along the path: 8 × 1 × 3 = 24.', gen: gen.chain3 },
        { t: 'mc', q: 'a = 3x and y = a². Which is dy/dx?', o: ['18x', '6x', '2a', '9x²'], ex: 'dy/da · da/dx = 2a · 3 = 6a, and a = 3x, so 18x. Stopping at 2a skips the da/dx link; 9x² is y itself.', gen: gen.chainSymbolic },
      ],
      cards: [
        { id: 'dv-chain', front: 'Chain rule on one path?', back: 'Multiply the local derivatives: dy/dx = dy/da · da/dx.' },
        { id: 'dv-chain-num', front: 'a = 2x, y = a², x = 3. dy/dx?', back: '2a · 2 = 12 × 2 = <b>24</b>.', gen: gen.cChain },
        { id: 'dv-chain-steps', front: 'Order of work for a chain?', back: 'Values forward, then local derivatives, then multiply along the path.' },
        { id: 'dv-chain-sym', front: 'a = 3x, y = a². dy/dx in terms of x?', back: '2a · 3 = <b>18x</b>.', gen: gen.cChainSym },
      ],
    },
    {
      id: 'dv-4',
      title: 'Computation graphs',
      steps: [
        { t: 'learn', title: 'Nodes and edges', body: '<p>A <b>computation graph</b> has one node per value and an edge from each input to the result.</p><div class="eq">x → a → b → y</div><p>a’s <b>parent</b> is x. a’s <b>child</b> is b. y’s <b>ancestors</b> are b, a and x: every node with a path to y.</p><p>The forward pass computes each value in order <b>and records the graph</b>, so the backward pass can retrace it.</p>' },
        { t: 'mc', q: 'In x → a → b → y, what are the parents of b?', o: ['a only', 'a and x', 'y', 'x only'], ex: 'A parent is a direct input. b is computed from a alone; x is an ancestor of b, not a parent.' },
        { t: 'mc', q: 'In x → a → b → y, which variables influence y?', o: ['x, a and b', 'b only', 'a and b only', 'x only'], ex: 'Influence flows along paths. x isn’t a direct parent of y, but changing x changes a, which changes b, which changes y.' },
        { t: 'mc', q: 'b affects d only through c. Can ∂d/∂b be non-zero even though b is not a direct parent of d?', o: ['Yes: indirect influence counts', 'No: only direct parents matter', 'Only if b is also a direct parent', 'Only if c is 0'], ex: 'A derivative measures total influence, through every path. b → c → d is a path, so ∂d/∂b = ∂d/∂c · ∂c/∂b, which is usually non-zero.' },
        { t: 'num', q: 'x → a → b → y with a = 2x, b = a + 5 and y = 3b. x is not a parent of y. What is ∂y/∂x at x = 4?', a: 6, tol: 0, ex: 'Not a parent, but there is a path: 2 × 1 × 3 = 6.', gen: gen.indirect },
        { t: 'match', q: 'Match each term to its meaning.', p: [['Parent of a', 'A node a is computed from'], ['Child of a', 'A node computed from a'], ['Ancestor of y', 'Any node with a path to y'], ['Forward pass', 'Computes values, records the graph']], ex: 'Parents and children are one edge apart; ancestors can be many edges back.' },
        { t: 'mc', q: 'What does the forward pass record, besides the values?', o: ['Which nodes each node came from, and by which operation', 'The gradients', 'Only the final output', 'Nothing else'], ex: 'The recorded graph is what the backward pass walks. Gradients come later, in the backward pass.' },
      ],
      cards: [
        { id: 'dv-parent', front: 'Parent vs child?', back: 'A node is computed from its parents. Its children are computed from it.' },
        { id: 'dv-ancestor', front: 'x → a → b → y. Does x influence y?', back: 'Yes. x is an ancestor, so ∂y/∂x can be non-zero.' },
        { id: 'dv-forward', front: 'What does the forward pass do?', back: 'Computes values and records the graph.' },
        { id: 'dv-indirect', front: 'b reaches d only through c. Can ∂d/∂b ≠ 0?', back: 'Yes.' },
      ],
    },
    {
      id: 'dv-5',
      title: 'Branches add up',
      steps: [
        { t: 'learn', title: 'Two roads from x', body: '<p>a = 2x, b = x², y = a + b. x reaches y along two paths: x → a → y and x → b → y. Multiply along each path, then <b>add</b> the results.</p><p>Path through a: dy/da · da/dx = 1·2. Path through b: dy/db · db/dx = 1·2x.</p><div class="eq">dy/dx = 2 + 2x</div><p>At x = 3: 2 + 6 = 8. Adding contributions into a running total is called <b>accumulating</b>.</p><p><b>Multiply along paths; add across paths.</b></p>' },
        { t: 'num', q: 'a = 2x, b = x² and y = a + b. At x = 3, what does the path through a contribute to dy/dx?', a: 2, tol: 0, ex: 'Along x → a → y: da/dx = 2 and dy/da = 1. Multiply: 2.', gen: gen.pathA },
        { t: 'num', q: 'a = 2x, b = x² and y = a + b. At x = 3, what does the path through b contribute to dy/dx?', a: 6, tol: 0, ex: 'Along x → b → y: db/dx = 2x = 6 and dy/db = 1. Multiply: 6.', gen: gen.pathB },
        { t: 'num', q: 'a = 2x, b = x² and y = a + b. At x = 3, what is the total dy/dx?', a: 8, tol: 0, ex: 'Through a: 2. Through b: 6. Add across paths: 8.', gen: gen.branchTotal },
        { t: 'mc', q: 'Two paths reach x. Do you overwrite or add their contributions?', o: ['Add them', 'Keep only the last one', 'Keep the larger', 'Multiply them'], ex: 'A nudge to x moves y through both paths at once, so the effects add. Accumulate means add.' },
        { t: 'num', q: 'a = 3x, b = 4x and y = a·b. At x = 2, what is dy/dx?', a: 48, tol: 0, ex: 'Through a: dy/da · da/dx = b · 3 = 8 × 3 = 24. Through b: a · 4 = 6 × 4 = 24. Add across paths: 48.', gen: gen.prodBranch },
        { t: 'mc', q: 'y = x·x. How many paths run from x to y?', o: ['Two: x feeds both inputs of the product', 'One', 'Zero', 'Four'], ex: 'Each input slot is its own path, each contributing x. They add to 2x: the derivative of x², rebuilt from the graph.' },
        { t: 'match', q: 'Match each phrase to what it means.', p: [['Along a path', 'Multiply'], ['Across paths', 'Add'], ['Accumulate', 'Add into the running total'], ['Overwrite', 'A bug: loses a path']], ex: 'Multiply along paths; add across paths.' },
      ],
      cards: [
        { id: 'dv-rule', front: 'Multiply along paths; …', back: 'add across paths.' },
        { id: 'dv-accumulate', front: 'Accumulate means?', back: 'Add.' },
        { id: 'dv-branch-num', front: 'a = 2x, b = x², y = a + b, x = 3. dy/dx?', back: '2 + 6 = <b>8</b>.', gen: gen.cBranch },
        { id: 'dv-xx', front: 'y = x·x as a graph: why is dy/dx = 2x?', back: 'Two paths, each contributing x, added.' },
        { id: 'dv-overwrite', front: 'Two paths reach x: overwrite or add?', back: 'Add.' },
      ],
    },
    {
      id: 'dv-6',
      title: 'Incoming gradient',
      steps: [
        { t: 'learn', title: 'Two numbers, one product', body: '<p>Working backward, each node receives an <b>incoming gradient</b> ∂L/∂(node): how much the final loss L moves when this node moves. To pass it on, multiply by the <b>local derivative</b>:</p><div class="eq hook">gradient sent to parent = incoming gradient × local derivative</div><p>c = a·b with a = 2, b = 3, and ∂L/∂c = 5:</p><ul><li>local ∂c/∂a = b = 3, so a receives 5 × 3 = <b>15</b></li><li>local ∂c/∂b = a = 2, so b receives 5 × 2 = <b>10</b></li></ul>' },
        { t: 'num', q: 'c = a·b with a = 2 and b = 3, and the incoming gradient is ∂L/∂c = 5. What gradient does a receive?', a: 15, tol: 0, ex: 'Local derivative ∂c/∂a = b = 3. Incoming × local: 5 × 3 = 15.', gen: gen.mulToA },
        { t: 'num', q: 'c = a·b with a = 2 and b = 3, and ∂L/∂c = 5. What gradient does b receive?', a: 10, tol: 0, ex: 'Local derivative ∂c/∂b = a = 2. Incoming × local: 5 × 2 = 10.', gen: gen.mulToB },
        { t: 'num', q: 'The incoming gradient is 5 and the local derivative is 3. What gets passed backward?', a: 15, tol: 0, ex: 'Incoming × local: 5 × 3 = 15. Not 5 (that ignores the local rule) and not 3 (that ignores everything downstream).', gen: gen.passBack },
        { t: 'mc', q: 'In ∂L/∂a = ∂L/∂c · ∂c/∂a, which factor is the local derivative?', o: ['∂c/∂a: it comes from c’s own operation', '∂L/∂c: it arrives from downstream', 'Both', 'Neither'], ex: '∂c/∂a only needs c’s operation and its inputs’ values. ∂L/∂c is the incoming gradient, built by everything between c and the loss.' },
        { t: 'num', q: 'c = a + b with a = 4 and b = −1, and ∂L/∂c = 7. What gradient does b receive?', a: 7, tol: 0, ex: '∂c/∂b = 1, so b gets 7 × 1 = 7. A sum hands the incoming gradient to each input unchanged.', gen: gen.sumTo },
        { t: 'num', q: 'c = x² with x = 4, and ∂L/∂c = 3. What gradient is sent to x?', a: 24, tol: 0, ex: 'Local: dc/dx = 2x = 8. Incoming × local: 3 × 8 = 24.', gen: gen.squareTo },
        { t: 'num', q: 'c = 4a and ∂L/∂c = 2.5. What gradient does a receive?', a: 10, tol: 0, ex: 'Local: ∂c/∂a = 4. Incoming × local: 2.5 × 4 = 10.', gen: gen.scaleTo },
        { t: 'mc', q: 'A node’s incoming gradient is 0. What does it send to its parents?', o: ['0, whatever its local derivatives are', 'Its local derivatives', '1', 'Its own value'], ex: 'Incoming × local = 0 × anything = 0. If the loss doesn’t care about this node, it doesn’t care about what fed it through this node.' },
      ],
      cards: [
        { id: 'dv-send', front: 'Gradient sent to a parent = ?', back: 'Incoming gradient × local derivative.' },
        { id: 'dv-send-num', front: 'Incoming 5, local 3. Passed back?', back: '<b>15</b>.', gen: gen.cSend },
        { id: 'dv-mul-back', front: 'c = a·b, a = 2, b = 3, ∂L/∂c = 5. Gradients to a and b?', back: 'a: 5 × 3 = <b>15</b>. b: 5 × 2 = <b>10</b>.', gen: gen.cMulBack },
        { id: 'dv-add-back', front: 'c = a + b, ∂L/∂c = 7. Gradients to a and b?', back: '<b>7</b> and <b>7</b>: a sum passes the gradient through unchanged.', gen: gen.cAddBack },
        { id: 'dv-local-incoming', front: 'Local derivative vs incoming gradient?', back: 'Local: ∂node/∂parent, from this node’s operation. Incoming: ∂L/∂node, from downstream.' },
      ],
    },
    {
      id: 'dv-7',
      title: 'Reverse mode',
      steps: [
        { t: 'learn', title: 'Start at the loss', body: '<p>Reverse-mode autodiff starts at the scalar loss with <span class="m">∂L/∂L = 1</span> and walks the graph backward. Once a node’s incoming gradient is complete, it sends incoming × local to each parent.</p><p>One backward sweep gives ∂L/∂(every node).</p>' },
        { t: 'mc', q: 'What gradient does the loss node start with?', o: ['∂L/∂L = 1', '0', 'The value of L', 'The learning rate'], ex: 'Nudging L by 0.01 changes L by 0.01. Everything downstream of that 1 is incoming × local.' },
        { t: 'num', q: 'a = x·w and L = (a − y)², with x = 2, w = 3 and y = 4. Forward: a = 6. Starting from ∂L/∂L = 1, what is ∂L/∂a?', a: 4, tol: 0, ex: 'The local derivative of (a − y)² with respect to a is 2(a − y) = 2 × (6 − 4) = 4. Times the incoming 1: 4.', gen: gen.lossA },
        { t: 'num', q: 'a = x·w and L = (a − y)², with x = 2, w = 3 and y = 4. What is ∂L/∂w?', a: 8, tol: 0, ex: 'Forward: a = 6. Backward: ∂L/∂a = 4. Then w gets incoming × local = 4 × x = 4 × 2 = 8.', gen: gen.lossW },
        { t: 'learn', title: 'Why reverse mode fits neural nets', body: '<p>Training has <b>one scalar loss</b> and <b>millions of parameters</b>. Reverse mode costs about one extra pass (a small constant times the forward pass) and returns ∂L/∂θ for every parameter at once.</p><p><b>Forward mode</b> pushes derivatives forward from one input per pass: ideal for few inputs and many outputs, hopeless for a million parameters.</p>' },
        { t: 'mc', q: 'Why is reverse mode the right choice for training a neural network?', o: ['One scalar loss, many parameters: one backward pass gives every gradient', 'It is more accurate than forward mode', 'It avoids the chain rule', 'It needs no forward pass'], ex: 'Both modes are exact. The difference is cost: reverse mode is one pass per output, forward mode one pass per input.' },
        { t: 'num', q: 'f maps 1,000,000 inputs to 1 output. How many reverse-mode passes give every derivative of every output?', a: 1, tol: 0, ex: 'Reverse mode gives the derivatives of one output with respect to every input per pass: 1 pass. Forward mode would take 1,000,000.', gen: gen.passes },
        { t: 'match', q: 'Match each term to its description.', p: [['Reverse mode', 'One pass per output'], ['Forward mode', 'One pass per input'], ['∂L/∂L', '1'], ['Backward pass', 'Computes sensitivities']], ex: 'Forward computes values. Backward computes sensitivities.' },
      ],
      cards: [
        { id: 'dv-seed', front: 'Reverse mode starts with ∂L/∂L = ?', back: '1.' },
        { id: 'dv-why-reverse', front: 'Why reverse mode for neural nets?', back: 'One scalar loss, many parameters: one backward pass gives every gradient.' },
        { id: 'dv-forward-mode', front: 'When does forward mode win?', back: 'Few inputs, many outputs.' },
        { id: 'dv-fwd-bwd', front: 'Forward computes …; backward computes …', back: 'Values; sensitivities.' },
        { id: 'dv-sq-loss', front: 'L = (a − y)², a = 6, y = 4. ∂L/∂a?', back: '2(a − y) = <b>4</b>.', gen: gen.cSqLoss },
      ],
    },
    {
      id: 'dv-8',
      title: 'A tiny autodiff engine',
      steps: [
        { t: 'learn', title: 'What a node stores', body: `<p>In a tiny engine (like micrograd) every node stores four things:</p>
<pre class="py">class Value:
    def __init__(self, data, parents=()):
        self.data = data           # value, from the forward pass
        self.grad = 0.0            # dL/d(this node), filled in backward
        self.parents = parents     # the nodes it was computed from
        self._backward = lambda: None   # local rule: push grad to parents

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += out.grad * other.data    # incoming x local
            other.grad += out.grad * self.data
        out._backward = _backward
        return out</pre><p>Note the <code>+=</code>.</p>` },
        { t: 'match', q: 'Match each field to what it holds.', p: [['data', 'The value from the forward pass'], ['grad', '∂L/∂node, starts at 0'], ['parents', 'Nodes it was computed from'], ['_backward', 'Sends incoming × local to each parent']], ex: 'Value, grad, parents and a local backward rule: that is the whole node.' },
        { t: 'mc', q: 'Why <code>self.grad += …</code> rather than <code>self.grad = …</code>?', o: ['A node used on several paths must add every contribution', 'It is faster', 'Python requires it', 'It keeps grad positive'], ex: 'Overwriting keeps only the last path’s contribution. Accumulating adds them all: add across paths.' },
        { t: 'num', q: 'Engine trace: a = x·x with x = 3 (x feeds both inputs) and L = a. After L.backward(), what is x.grad?', a: 6, tol: 0, ex: 'The backward rule runs once per input slot: x.grad += 1 × 3, twice. Accumulate: 3 + 3 = 6 = d(x²)/dx.', gen: gen.engineXX },
        { t: 'num', q: 'Same graph, a = x·x with x = 3 and L = a, but the backward rule overwrites: <code>x.grad = out.grad * other.data</code>. What does x.grad end up as?', a: 3, tol: 0, ex: 'Both writes set 3; the second replaces the first, leaving 3. The true dL/dx is 6: overwriting lost a path.', gen: gen.engineOverwrite },
        { t: 'learn', title: 'Visit nodes in the right order', body: `<p><code>L.backward()</code> sets <code>L.grad = 1</code>, then calls each node’s <code>_backward</code> in <b>reverse topological order</b>: a node runs only after every child has added into its grad.</p><pre class="py">order = topological_sort(L)    # parents before children
L.grad = 1.0
for node in reversed(order):   # children before parents
    node._backward()</pre>` },
        { t: 'mc', q: 'Why reverse topological order?', o: ['A node’s grad must be complete (every child added) before it passes it on', 'It visits nodes alphabetically', 'It saves memory', 'Any order gives the same result'], ex: 'If a node sent its grad to its parents before a second child had added in, the parents would miss that contribution.' },
        { t: 'num', q: 'c = a·b and L = c + a, with a = 2 and b = 5. After L.backward(), what is a.grad?', a: 6, tol: 0, ex: 'Two paths into a. Through c: 1 × b = 5. Directly into L: 1. Accumulate: 5 + 1 = 6.', gen: gen.engineMix },
        { t: 'mc', q: 'Why reset every grad to 0 before each backward pass?', o: ['Grads accumulate, so last step’s values would add in', 'To free memory', 'The forward pass needs it', 'It isn’t necessary'], ex: 'Accumulation is right within one backward pass and wrong across two. That’s what <code>optimizer.zero_grad()</code> is for in PyTorch.' },
      ],
      cards: [
        { id: 'dv-node', front: 'What does an autodiff node store?', back: 'Value, grad, parents, and a local backward rule.' },
        { id: 'dv-plus-eq', front: 'Why grad += rather than = ?', back: 'A node on several paths must add every contribution.' },
        { id: 'dv-topo', front: 'Order of the backward pass?', back: 'Reverse topological: each node after all its children.' },
        { id: 'dv-zero', front: 'Why zero the grads before backward?', back: 'They accumulate, so old values would add in.' },
        { id: 'dv-engine-xx', front: 'a = x·x, x = 3, L = a. x.grad after backward?', back: '<b>6</b>: 3 + 3, one per input slot.', gen: gen.cEngineXX },
      ],
    },
    {
      id: 'dv-9',
      title: 'Explain backprop',
      steps: [
        { t: 'learn', title: 'Say it out loud', body: '<p>These are interview prompts. Answer first, out loud or in writing, then tick what you covered. Aim to use the three hooks:</p><div class="eq hook">incoming gradient × local derivative</div><div class="eq hook">multiply along paths; add across paths</div><div class="eq hook">forward computes values; backward computes sensitivities</div>' },
        { t: 'explain', kind: 'recall', secs: 30, q: 'Explain backprop in 30 seconds.', model: '<p>Backprop is reverse-mode autodiff on the computation graph. The <b>forward pass</b> computes every value and records the graph. The <b>backward pass</b> starts at the scalar loss with ∂L/∂L = 1 and visits nodes in reverse topological order. Each node multiplies its <b>incoming gradient</b> by its <b>local derivative</b> to get each parent’s share: that’s the chain rule. When a node feeds several paths, the contributions <b>add</b>. One backward pass gives ∂L/∂θ for every parameter.</p>', rubric: ['Forward pass computes values and records the graph', 'Starts from the scalar loss with ∂L/∂L = 1 and goes backward', 'Gradient to a parent = incoming gradient × local derivative (the chain rule)', 'Contributions from several paths are added (accumulated)', 'One backward pass gives every parameter’s gradient'] },
        { t: 'explain', kind: 'derive', secs: 60, q: 'Why do gradients add when a variable affects the loss through several paths?', model: '<p>∂L/∂x is the total change in L per small change in x. If x reaches L along two paths, a nudge to x moves L through both at once, and to first order the two effects add. That is the multivariable chain rule: ∂L/∂x = Σ over x’s children of ∂L/∂child · ∂child/∂x.</p><p>Example: y = x·x has two paths, each contributing x, total 2x. Overwriting instead of adding would drop a path, which is why engines write <code>grad +=</code>.</p>', rubric: ['A small change in x reaches L through every path at once', 'To first order the effects add: a sum over paths of the product along each path', 'Names the multivariable chain rule or the sum over children', 'Gives an example such as x·x → 2x', 'Overwriting would lose a path, hence +='] },
        { t: 'explain', kind: 'defend', secs: 60, q: 'Why is reverse-mode autodiff the right tool for neural networks?', model: '<p>Training needs ∂L/∂θ for millions of parameters, but L is a <b>single scalar</b>. Reverse mode gives the derivative of one output with respect to all inputs in one backward pass, at a small constant times the cost of the forward pass. Forward mode gives derivatives with respect to one input per pass, so it would need a pass per parameter; it only wins with few inputs and many outputs. The price of reverse mode is memory: the forward pass must keep the values the backward pass needs.</p>', rubric: ['One scalar loss, many parameters', 'One backward pass gives every parameter’s gradient, at a constant times the forward cost', 'Forward mode needs one pass per input or parameter', 'Forward mode suits few inputs and many outputs', 'Mentions the memory cost of storing forward values'] },
        { t: 'explain', kind: 'recall', secs: 45, q: 'What does an autodiff node need to store, and why?', model: '<p>Its <b>value</b> from the forward pass (local derivatives often need it, e.g. ∂(a·b)/∂a = b). Its <b>grad</b>, ∂L/∂node, starting at 0 and accumulated with +=. Its <b>parents</b>, the graph edges the backward pass retraces. And its operation’s <b>local backward rule</b>, which turns its grad into each parent’s contribution: incoming gradient × local derivative.</p>', rubric: ['The value, because local derivatives use it', 'The grad, starting at 0 and accumulated', 'The parents (the recorded graph)', 'The operation’s local backward rule', 'The rule computes incoming gradient × local derivative'] },
        { t: 'mc', q: 'Which sentence describes the backward pass?', o: ['It computes sensitivities: ∂L/∂(each node)', 'It recomputes the values', 'It updates the weights', 'It computes the loss'], ex: 'Forward computes values; backward computes sensitivities. Updating the weights is the optimizer’s job, afterwards.' },
      ],
      cards: [
        { id: 'dv-hooks', front: 'The three backprop hooks?', back: 'Incoming gradient × local derivative. Multiply along paths; add across paths. Forward computes values; backward computes sensitivities.' },
        { id: 'dv-say-backprop', kind: 'recall', secs: 30, front: 'Explain backprop in 30 seconds.', back: 'Forward: values and graph. Backward from ∂L/∂L = 1 in reverse topological order: each node sends incoming × local to its parents, and contributions add. One pass, every gradient.', rubric: ['Forward computes values and records the graph', 'Backward starts at ∂L/∂L = 1', 'Incoming × local (chain rule)', 'Paths add'] },
        { id: 'dv-say-add', kind: 'derive', secs: 45, front: 'Why do gradients add across paths?', back: 'A nudge to x moves L through every path at once; to first order the effects add (the multivariable chain rule).', rubric: ['Every path carries part of the change', 'Effects add to first order', 'Example such as x·x → 2x'] },
        { id: 'dv-say-reverse', kind: 'defend', secs: 45, front: 'Why reverse mode for neural nets?', back: 'One scalar loss, many parameters: one backward pass gives them all. Forward mode needs a pass per parameter.', rubric: ['One scalar output, many parameters', 'One backward pass for all gradients', 'Forward mode: one pass per input'] },
      ],
    },
    {
      id: 'dv-10',
      title: 'Vectors & Jacobians',
      steps: [
        { t: 'learn', title: 'Many outputs', body: '<p>When f maps n inputs to m outputs, its local derivative is a table: the <b>Jacobian</b> J, with shape <b>(m, n)</b> and</p><div class="eq">J[i, j] = ∂yᵢ/∂xⱼ</div><p>Row i shows how output i responds to each input. For y = W x, J = W.</p>' },
        { t: 'num', q: 'f maps 4 inputs to 3 outputs. How many entries does its Jacobian have?', a: 12, tol: 0, ex: 'J has shape (outputs, inputs) = (3, 4): 12 entries.', gen: gen.jacSize },
        { t: 'num', q: 'y = W x with W = {{2 −1 0; 1 3 4}}. What is ∂y₀/∂x₁?', a: -1, tol: 0, ex: 'y₀ = 2·x₀ + (−1)·x₁ + 0·x₂, so ∂y₀/∂x₁ = W[0, 1] = −1. The Jacobian of Wx is W itself.', gen: gen.jacEntry },
        { t: 'mc', q: 'For y = W x, what is the Jacobian ∂y/∂x?', o: ['W', 'Wᵀ', 'x', 'xᵀ'], ex: 'y is linear in x with coefficients W, so J[i, j] = W[i, j].' },
        { t: 'learn', title: 'Vector-Jacobian products', body: '<p>Backprop never needs the whole Jacobian. A node receives the incoming gradient <b>g = ∂L/∂y</b> (shaped like y) and sends back <b>gᵀJ</b> (shaped like x): the vector version of incoming × local, called a <b>vector-Jacobian product</b>.</p><div class="eq">y = W x  →  ∂L/∂x = Wᵀ g</div>' },
        { t: 'num', q: 'y = W x with W = {{1 2; 3 4}} and incoming gradient g = ∂L/∂y = [1, −1]. What is entry 0 of ∂L/∂x = Wᵀg?', a: -2, tol: 0, ex: 'Entry 0 is column 0 of W dotted with g: 1·1 + 3·(−1) = −2.', gen: gen.vjp },
        { t: 'mc', q: 'Why doesn’t backprop build the full Jacobian?', o: ['It only needs gᵀJ, which is far smaller and can be computed directly', 'Jacobians are always zero', 'The chain rule doesn’t apply to vectors', 'It does build it'], ex: 'A layer with 4,096 inputs and outputs has a 16.7-million-entry Jacobian per example; gᵀJ has 4,096 entries. For y = Wx, Wᵀg is one matrix–vector product.' },
        { t: 'num', q: 'An elementwise operation (like ReLU) maps 1,000 values to 1,000 values. How many entries does its full Jacobian have?', a: 1000000, tol: 0, ex: '1,000 × 1,000 = 1,000,000. Only the 1,000 diagonal entries can be non-zero, so backprop just multiplies g elementwise by f′(x).', gen: gen.elemJac },
        { t: 'match', q: 'Match each idea to its meaning.', p: [['Jacobian', '(outputs, inputs) table of ∂yᵢ/∂xⱼ'], ['VJP', 'gᵀJ: incoming gradient times Jacobian'], ['y = Wx backward', '∂L/∂x = Wᵀg'], ['Elementwise op', 'Diagonal Jacobian: multiply elementwise']], ex: 'Same rule as the scalar case, incoming × local, with vectors in place of numbers.' },
      ],
      cards: [
        { id: 'dv-jac', front: 'Jacobian of f: 4 inputs → 3 outputs. Shape?', back: '<b>(3, 4)</b>: J[i, j] = ∂yᵢ/∂xⱼ, outputs by inputs.', gen: gen.cJac },
        { id: 'dv-jac-linear', front: 'Jacobian of y = Wx?', back: 'W.' },
        { id: 'dv-vjp', front: 'In vector backprop, what does a node send back?', back: 'gᵀJ, the vector-Jacobian product, shaped like its input.' },
        { id: 'dv-wtg', front: 'y = Wx, incoming g. ∂L/∂x?', back: 'Wᵀg.' },
        { id: 'dv-no-jac', front: 'Why avoid forming the full Jacobian?', back: 'It’s huge, and gᵀJ is all backprop needs.' },
      ],
    },
  ],
};
