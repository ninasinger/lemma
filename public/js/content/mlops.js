// Unit: ML CI/CD. Promotion contracts, slice gates, small-sample uncertainty, leakage vs
// training-serving skew, and safe rollout. Mostly judgment, so most drills are scenario
// questions drawn from pools; the two number drills make the small-sample point concrete.
import { num, big } from '../vary.js';

// Fresh scenarios every time a question is shown (see ../vary.js).
const LAYER = { sw: 'Software and systems correctness', serve: 'Serving and performance', ml: 'ML behavior' };
const LAYER_EXAMPLES = [
  ['The API still returns the agreed JSON schema', 'sw'], ['A saved model loads back with identical weights', 'sw'],
  ['Predictions are written to the database correctly', 'sw'], ['A dependency upgrade doesn’t break the service', 'sw'],
  ['Malformed requests are rejected cleanly', 'sw'], ['p99 latency stays under 80 ms at peak load', 'serve'],
  ['The model fits in 16 GB of GPU memory', 'serve'], ['A cold start takes under 30 seconds', 'serve'],
  ['The service sustains 2,000 requests per second', 'serve'], ['No out-of-memory crashes under load', 'serve'],
  ['Accuracy beats the current production model', 'ml'], ['Recall on the self-harm slice stays above its threshold', 'ml'],
  ['A fixed set of sanity examples still gets the right labels', 'ml'], ['Confidence scores stay calibrated', 'ml'],
  ['Safety guardrails still fire on known bad inputs', 'ml'],
];
const SAFETY = ['recall on the self-harm slice drops below its pre-agreed threshold', 'the threat-detection slice fails its hard gate',
  'violence recall regresses past its safety limit', 'the high-severity slice falls below its safety threshold'];
const BUSINESS = ['accuracy on a low-traffic, low-impact product category dips 0.3 points while the main KPI rises 4%',
  'quality on short, low-stakes queries dips slightly while the primary metric improves substantially',
  'a cosmetic formatting slice regresses a little while revenue per session improves',
  'a low-impact language slice dips within its noise while overall quality improves'];
const SCENARIOS = [
  ['A churn model uses “customer canceled within the next 7 days” to predict cancellation tomorrow', 'leak'],
  ['A fraud model’s feature is computed from the chargeback that arrives weeks after the transaction', 'leak'],
  ['Normalization statistics are fit on the training and test sets together before evaluation', 'leak'],
  ['The same users appear in both the training and the evaluation set', 'leak'],
  ['A feature uses the account’s final status, recorded after the prediction date', 'leak'],
  ['Training counts transactions in the 24 hours before prediction; production counts the previous calendar day', 'skew'],
  ['Training normalizes with fixed training-set mean and std; serving recomputes them from each batch', 'skew'],
  ['Serving uses a newer tokenizer version than training did', 'skew'],
  ['Missing values become 0 in training but −1 in the serving pipeline', 'skew'],
  ['The serving code passes features in a different column order than training', 'skew'],
  ['Training timestamps are in UTC; production computes the same feature in local time', 'skew'],
];
const KIND_LABEL = { leak: 'Data leakage', skew: 'Training-serving skew' };
const KIND_EX = {
  leak: 'The model used information it couldn’t legitimately have at prediction time, so its offline numbers are too good to be true.',
  skew: 'The feature means something different in production than it did in training: the model sees inputs unlike the ones it learned from.',
};
const SYSTEM = ['p99 latency', 'error rate', 'GPU memory use', 'requests per second', 'out-of-memory crashes', 'CPU saturation'];
const PRODUCT = ['recall on the self-harm slice', 'click-through rate', 'user complaints', 'calibration error', 'the primary quality metric', 'how often safety guardrails fire'];
const STAGES = {
  offline: ['Metrics, slices and parity tests run on held-out data before any traffic', 'Offline gates'],
  shadow: ['The candidate gets real production inputs, but users never see its outputs', 'Shadow deployment'],
  canary: ['5% of users get the candidate while everyone else stays on production', 'Canary'],
  ramp: ['Traffic moves 10% → 25% → 50% as the monitored metrics hold', 'Gradual ramp'],
  rollback: ['The previous immutable model version takes all traffic again', 'Rollback'],
};
const half = (n, p) => 196 * Math.sqrt((p * (1 - p)) / n);

const gen = {
  layerOf(v) {
    const [ex, k] = v.pick(LAYER_EXAMPLES);
    return { q: `Which layer of ML CI/CD checks this: “${ex}”?`, o: [LAYER[k], ...Object.keys(LAYER).filter((x) => x !== k).map((x) => LAYER[x]), 'None: CI can’t check that'],
      ex: { sw: 'Does the system work? Contracts, serialization, writes, dependencies, failure handling.', serve: 'Can it serve? Latency, throughput, memory, startup, behavior under load.', ml: 'Does the model behave acceptably? Metrics, slices, sanity sets, calibration, guardrails.' }[k] };
  },
  gate(v) {
    const safety = v.int(0, 1) === 1, s = v.pick(safety ? SAFETY : BUSINESS);
    return safety
      ? { q: `A candidate improves overall, but ${s}. What should happen?`, o: ['Block: a safety gate fails closed', 'Promote: the overall gain outweighs it', 'Let the researcher override it', 'Promote now and fix it in the next model'],
        ex: 'Hard safety gates fail closed. If an exception is ever possible, it goes through a formal risk-acceptance process, not an ad hoc override.' }
      : { q: `A candidate improves overall, but ${s}. What should happen?`, o: ['Weigh it explicitly (impact, size, uncertainty, governance), then document an exception or recalibrate the gate', 'Block automatically, as for a safety gate', 'Promote without recording anything', 'Drop the slice from the contract'],
        ex: 'A business gate is an explicit, evidence-backed tradeoff. Write the decision down, or fix the gate if it was miscalibrated.' };
  },
  ciHalf(v) {
    const n = v.pick([100, 200, 400, 1000, 5000]), p = v.pick([0.8, 0.9, 0.95]), h = half(n, p);
    return { q: `A critical slice has ${big(n)} examples and ${num(100 * p)}% accuracy. About how many percentage points either side is a 95% interval?`, a: +h.toFixed(2),
      ex: `±1.96 × √(p(1 − p) / n) = ±1.96 × √(${num(p)} × ${num(1 - p)} / ${big(n)}) ≈ ±${num(h, 1)} points. Quadruple the examples to halve it.`,
      py: `ans = 1.96 * 100 * (rng.binomial(${n}, ${p}, 400000) / ${n}).std()` };
  },
  sampleSize(v) {
    const p = v.pick([0.8, 0.9, 0.95]), h = v.pick([1, 2, 3, 5]), n = Math.ceil((1.96 ** 2 * p * (1 - p)) / (h / 100) ** 2);
    return { q: `A slice runs at about ${num(100 * p)}% accuracy. Roughly how many examples give a 95% interval of ±${h} points?`, a: n, tol: Math.max(2, Math.round(0.02 * n)),
      ex: `n = 1.96² × p(1 − p) / h² = 3.84 × ${num(p * (1 - p), 4)} / ${num(h / 100)}² ≈ ${big(n)}.`,
      py: `n = 1\nwhile 1.96 * np.sqrt(${p} * (1 - ${p}) / n) > ${h / 100}:\n    n += 1\nans = n` };
  },
  noise(v) {
    const n = v.pick([100, 200, 400]), p = v.pick([0.8, 0.9]), h = half(n, p), real = v.int(0, 1) === 1;
    const drop = real ? Math.ceil(1.7 * h) : Math.max(1, Math.floor(0.5 * h)), pts = `${drop} point${drop === 1 ? '' : 's'}`;
    const yes = 'It’s unlikely to be noise: treat it as a real regression, after checking for duplicates and representativeness';
    const maybe = 'It could easily be noise: get more examples before concluding (or block, if the slice is safety-critical)';
    return { q: `On ${n} slice examples the canary scores ${pts} below production, around ${num(100 * p)}% accuracy. Production’s number comes from far more traffic. The canary’s 95% interval is about ±${num(h, 1)} points. What can you conclude?`,
      o: [real ? yes : maybe, real ? maybe : yes, 'The canary is actually better', 'Intervals don’t apply to slices'],
      ex: real ? `${pts} is well outside ±${num(h, 1)}, so chance alone is an unlikely explanation.` : `${pts} is well inside ±${num(h, 1)}: a sample this size moves that much by chance.` };
  },
  leakOrSkew(v) {
    const [s, k] = v.pick(SCENARIOS);
    return { q: `${s}. What is this?`, o: [KIND_LABEL[k], KIND_LABEL[k === 'leak' ? 'skew' : 'leak'], 'Neither: this is fine'], ex: KIND_EX[k] };
  },
  metricKind(v) {
    const system = v.int(0, 1) === 1, m = v.pick(system ? SYSTEM : PRODUCT);
    return { q: `During a canary you watch ${m}. What kind of signal is it?`, o: [system ? 'A system metric' : 'A model or product metric', system ? 'A model or product metric' : 'A system metric', 'Neither: not worth watching'],
      ex: system ? 'System metrics ask whether the service is healthy: latency, throughput, errors, OOMs, resources.' : 'Model and product metrics ask whether it behaves: quality, critical slices, KPIs, guardrails, complaints, calibration.' };
  },
  stage(v) {
    const keys = Object.keys(STAGES), k = v.pick(keys);
    return { q: `“${STAGES[k][0]}.” Which stage is this?`, o: [STAGES[k][1], ...keys.filter((x) => x !== k).map((x) => STAGES[x][1]).slice(0, 3)],
      ex: 'The path is offline gates → shadow → canary → gradual ramp → full production, with rollback ready at every step.' };
  },
};

export default {
  id: 'ops',
  title: 'ML CI/CD',
  sub: 'Testing the model, not just the code: promotion gates, skew and safe rollouts.',
  c: 'ops',
  guide: `<h3>Memory hooks</h3><div class="eq hook">Does the system work? Does the model behave?</div><div class="eq hook">Humans define the contract; the pipeline enforces it. Set the rules before seeing the candidate.</div><div class="eq hook">Slices reflect failure cost.</div><div class="eq hook">Safety gate = fail closed. Business gate = explicit, evidence-backed tradeoff.</div><div class="eq hook">Small slices need uncertainty, not point estimates.</div><div class="eq hook">Leakage = saw what it shouldn’t. Skew = sees something different in production.</div><div class="eq hook">Same raw input → same model feature.</div><div class="eq hook">Promotion without rollback is incomplete.</div>
  <h3>Three layers of tests</h3><ul><li><b>Software and systems:</b> API contracts, serialization, database writes, dependencies, schema validation, failure handling.</li><li><b>Serving and performance:</b> memory, latency, throughput, startup time, CPU/GPU use, OOM behavior, load tests.</li><li><b>ML behavior:</b> primary metric vs production, critical slices, sanity and regression sets, calibration, product and safety guardrails.</li></ul>
  <h3>Promotion contract</h3><ul><li>Defined in advance: metric floor, max regression vs production, critical-slice thresholds, minimum sample size or confidence, latency/throughput SLOs, memory limits, preprocessing parity, provenance, rollout and rollback criteria.</li></ul>
  <h3>Slices and gates</h3><ul><li>Aggregate metrics can improve while an important slice regresses badly.</li><li>Pick slices by failure cost: user harm, business, contractual, regulatory, safety.</li><li>Safety gates fail closed; any exception needs formal risk acceptance.</li><li>Business gates weigh impact, magnitude, confidence, gains elsewhere and constraints, and may expose a miscalibrated gate.</li></ul>
  <h3>Small samples</h3><ul><li>95% interval on an accuracy p from n examples: about ±1.96·√(p(1 − p)/n). 200 examples at 90%: ±4.2 points.</li><li>Check independence, duplicates, representativeness and the traffic being compared. For a safety-critical slice, uncertainty itself can block.</li><li>Contracts can require a minimum count, confidence intervals, significance or equivalence tests, and good labels.</li></ul>
  <h3>Leakage vs skew</h3><ul><li><b>Leakage:</b> training or evaluation used information not available at prediction time (“canceled within the next 7 days”).</li><li><b>Skew:</b> the same feature is computed differently in training and production (24 hours before vs previous calendar day; fixed training mean/std vs per-batch stats).</li><li>Prevent skew with versioned preprocessing tied to the model, shared transformation code, schema/order/missing-value/tokenization/normalization tests, and offline/online parity tests.</li></ul>
  <h3>Rollout</h3><ul><li>Offline gates → shadow → canary → gradual ramp → full production.</li><li>Watch system metrics (latency, throughput, errors, OOMs, resources) and model/product metrics (quality, slices, KPIs, guardrails, complaints, calibration).</li><li>Define rollback in advance: triggers, who or what fires it, how fast, and which immutable model version and serving/preprocessing config come back.</li></ul>`,
  lessons: [
    {
      id: 'ops-1',
      title: 'Three layers of tests',
      steps: [
        { t: 'learn', title: 'More than “does it run?”', body: '<p>Ordinary CI/CD asks whether the code runs, unit tests pass, API contracts hold and services talk to each other. ML CI/CD needs all of that <b>plus the behavior of the model</b>:</p><ul><li><b>Software and systems:</b> API contracts, serialization, database writes, dependencies, schemas, failure handling.</li><li><b>Serving and performance:</b> memory, latency, throughput, startup, CPU/GPU use, OOMs, load.</li><li><b>ML behavior:</b> primary metric vs production, critical slices, sanity sets, calibration, guardrails.</li></ul><div class="eq hook">Does the system work? Does the model behave acceptably?</div>' },
        { t: 'mc', q: 'Which layer of ML CI/CD checks this: “p99 latency stays under 80 ms at peak load”?', o: ['Serving and performance', 'Software and systems correctness', 'ML behavior', 'None: CI can’t check that'], ex: 'Can it serve? Latency, throughput, memory, startup, behavior under load.', gen: gen.layerOf },
        { t: 'mc', q: 'Which layer of ML CI/CD checks this: “Recall on the self-harm slice stays above its threshold”?', o: ['ML behavior', 'Software and systems correctness', 'Serving and performance', 'None: CI can’t check that'], ex: 'Does the model behave acceptably? Metrics, slices, sanity sets, calibration, guardrails.', gen: gen.layerOf },
        { t: 'tf', q: 'If every unit and integration test passes, the model is safe to ship.', a: false, ex: 'Those prove the system works. A model can pass all of them and still regress on a critical slice, lose calibration or break a guardrail.' },
        { t: 'mc', q: 'Which layer of ML CI/CD checks this: “A saved model loads back with identical weights”?', o: ['Software and systems correctness', 'Serving and performance', 'ML behavior', 'None: CI can’t check that'], ex: 'Does the system work? Contracts, serialization, writes, dependencies, failure handling.', gen: gen.layerOf },
        { t: 'match', q: 'Match each check to its layer.', p: [['API contract', 'Software and systems'], ['Memory footprint', 'Serving and performance'], ['Critical-slice recall', 'ML behavior'], ['Fixed sanity examples', 'Regression set for behavior']], ex: 'Two layers ask whether the system works; the third asks whether the model behaves.' },
      ],
      cards: [
        { id: 'ops-layers', front: 'Which test layer checks latency and memory?', back: 'Serving and performance.' },
        { id: 'ops-extra', front: 'What does ML CI/CD check that ordinary CI/CD doesn’t?', back: 'Whether the model behaves acceptably.' },
        { id: 'ops-serving', front: 'Which test layer checks API contracts?', back: 'Software and systems correctness.' },
        { id: 'ops-behavior', front: 'Which test layer checks critical-slice metrics?', back: 'ML behavior.' },
      ],
    },
    {
      id: 'ops-2',
      title: 'Promotion contracts',
      steps: [
        { t: 'learn', title: 'Decide before you look', body: '<p>Don’t promote a model on a judgment call at launch time. Define a <b>promotion contract</b> in advance:</p><ul><li>minimum primary metric and maximum regression vs production</li><li>hard thresholds on critical slices, with a minimum sample size or confidence</li><li>latency, throughput and memory limits</li><li>preprocessing parity and provenance</li><li>rollout and rollback criteria</li></ul><div class="eq hook">Humans define the contract; the pipeline enforces it consistently.</div><div class="eq hook">Set the rules before seeing the candidate.</div>' },
        { t: 'mc', q: 'When should a model’s promotion thresholds be set?', o: ['Before seeing the candidate model', 'After seeing its offline results', 'At the launch meeting', 'After a week in production'], ex: 'Rules written after the results are in tend to bend toward the model you’ve already invested in.' },
        { t: 'mc', q: '“Humans define the contract…”', o: ['…and the pipeline enforces it consistently', '…and humans re-decide it at each launch', '…and the model sets its own thresholds', '…and nobody enforces it'], ex: 'People decide what acceptable means; automation applies it the same way every time.' },
        { t: 'mc', q: 'Which of these belongs in a promotion contract?', o: ['A hard threshold on a critical slice', 'The researcher’s confidence that it’s better', 'Whatever the launch meeting decides', 'A smooth-looking training-loss curve'], ex: 'Contract items are measurable and agreed in advance. Confidence, meetings and nice curves are not gates.' },
        { t: 'match', q: 'Match each contract item to what it guards.', p: [['Minimum primary metric', 'A quality floor'], ['Max regression vs production', 'Backsliding'], ['Latency and memory limits', 'Serving SLOs'], ['Rollback criteria', 'The exit plan']], ex: 'A contract covers quality, behavior, serving and the way out.' },
        { t: 'explain', kind: 'recall', secs: 60, q: 'What goes into a model promotion contract, and why is it set in advance?', model: '<p>A promotion contract lists what a candidate must meet: a minimum primary metric, a maximum regression against production, hard thresholds on critical slices with a minimum sample size or confidence requirement, latency/throughput SLOs and memory limits, preprocessing parity and provenance, and the rollout and rollback criteria. Humans agree on it <b>before</b> seeing the candidate and the pipeline enforces it, so teams can’t rationalize a regression after investing in a new model, and every launch is judged the same way.</p>', rubric: ['Quality: metric floor and max regression vs production', 'Critical-slice thresholds with sample-size or confidence requirements', 'Serving limits: latency, throughput, memory', 'Parity, provenance, rollout and rollback criteria', 'Set in advance and enforced by the pipeline, to prevent rationalizing regressions'] },
      ],
      cards: [
        { id: 'ops-contract', front: 'What is a promotion contract?', back: 'Pre-agreed criteria a model must meet to ship.' },
        { id: 'ops-contract-when', front: 'When is a promotion contract set?', back: 'Before seeing the candidate model.' },
        { id: 'ops-contract-who', front: 'Who enforces a promotion contract?', back: 'The pipeline, the same way every time.' },
        { id: 'ops-contract-items', front: 'Why set the contract in advance?', back: 'So nobody rationalizes a regression after the fact.' },
      ],
    },
    {
      id: 'ops-3',
      title: 'Slices and gates',
      steps: [
        { t: 'learn', title: 'Averages hide failures', body: '<p>A moderation model can improve overall while regressing badly on self-harm, violence, threats, sexual content, low-resource languages or high-severity borderline cases.</p><div class="eq hook">Slices reflect failure cost, not convenient metadata.</div><p>A slice deserves a gate when a regression there would cause disproportionate user harm, business impact, contractual risk, regulatory exposure or safety risk.</p>' },
        { t: 'mc', q: 'Overall accuracy improves by 2 points. Why might that not be enough to promote?', o: ['An important slice can regress badly inside an improving average', 'Accuracy can’t go up by 2 points', 'Aggregate metrics are always wrong', 'Promotion only depends on latency'], ex: 'Large, easy slices dominate an average. A small, high-stakes slice can get much worse without moving it.' },
        { t: 'tf', q: 'Critical slices should be whatever metadata is easiest to log.', a: false, ex: 'Choose them by failure cost: where a regression would hurt users, the business, contracts, regulators or safety.' },
        { t: 'learn', title: 'Two kinds of gate', body: '<p><b>Safety-critical gates fail closed.</b> Violating a pre-agreed safety threshold blocks promotion. If exceptions exist at all, they need a formal risk-acceptance process, not a researcher override.</p><p><b>Business gates are explicit tradeoffs.</b> A small regression on a low-impact slice can be acceptable if the benefit is large, weighing the slice’s impact, the size of the regression, statistical confidence, gains elsewhere, contractual or fairness constraints, and whether the gate itself is miscalibrated.</p><div class="eq hook">Safety gate = fail closed. Business gate = explicit, evidence-backed tradeoff.</div>' },
        { t: 'mc', q: 'A candidate improves overall, but recall on the self-harm slice drops below its pre-agreed threshold. What should happen?', o: ['Block: a safety gate fails closed', 'Promote: the overall gain outweighs it', 'Let the researcher override it', 'Promote now and fix it in the next model'], ex: 'Hard safety gates fail closed. If an exception is ever possible, it goes through a formal risk-acceptance process, not an ad hoc override.', gen: gen.gate },
        { t: 'mc', q: 'A candidate improves overall, but accuracy on a low-traffic, low-impact product category dips 0.3 points while the main KPI rises 4%. What should happen?', o: ['Weigh it explicitly (impact, size, uncertainty, governance), then document an exception or recalibrate the gate', 'Block automatically, as for a safety gate', 'Promote without recording anything', 'Drop the slice from the contract'], ex: 'A business gate is an explicit, evidence-backed tradeoff. Write the decision down, or fix the gate if it was miscalibrated.', gen: gen.gate },
        { t: 'mc', q: 'If an exception to a safety gate is ever allowed, how should it happen?', o: ['Through a formal risk-acceptance process', 'By the researcher who trained the model', 'By lowering the threshold quietly', 'By rerunning the evaluation until it passes'], ex: 'Exceptions to safety gates need accountable sign-off and a record, not a judgment call by the people invested in the model.' },
        { t: 'mc', q: 'A candidate improves overall, but the threat-detection slice fails its hard gate. What should happen?', o: ['Block: a safety gate fails closed', 'Promote: the overall gain outweighs it', 'Let the researcher override it', 'Promote now and fix it in the next model'], ex: 'Hard safety gates fail closed.', gen: gen.gate },
        { t: 'explain', kind: 'defend', secs: 60, q: 'A candidate slightly regresses on a low-impact, non-safety slice but substantially improves a primary business KPI. What do you do?', model: '<p>Treat it as a business gate, not a safety gate. Weigh the slice’s real impact, the size of the regression and whether it is statistically real, the size of the gain elsewhere, and any contractual or fairness constraints. If the tradeoff is acceptable, promote with a documented, formal exception, or recalibrate the gate if it was set wrong. Either way the decision is explicit and on record, not a quiet override.</p>', rubric: ['Identifies it as a business (non-safety) gate', 'Weighs impact, magnitude and uncertainty of the regression', 'Considers contractual or fairness constraints', 'Documents a formal exception or recalibrates the gate'] },
      ],
      cards: [
        { id: 'ops-aggregate', front: 'Why aren’t aggregate metrics enough?', back: 'An important slice can regress inside an improving average.' },
        { id: 'ops-slice-choice', front: 'How should critical slices be chosen?', back: 'By the cost of failure.' },
        { id: 'ops-safety-gate', front: 'A safety gate fails. What happens?', back: 'Promotion is blocked: it fails closed.' },
        { id: 'ops-business-gate', front: 'A small regression on a low-impact slice?', back: 'An explicit, documented tradeoff.' },
      ],
    },
    {
      id: 'ops-4',
      title: 'Small samples need uncertainty',
      steps: [
        { t: 'learn', title: '200 examples, 8 points worse', body: '<p>A canary has only 200 examples from a critical slice and looks 8 points worse than production. Don’t trust the point estimate blindly. Ask:</p><ul><li>Are the examples independent? Any duplicates?</li><li>Is the slice representative, and is it the same traffic distribution as production’s?</li><li>How big is the uncertainty?</li><li>Is the slice so safety-critical that uncertainty itself should block?</li></ul><p>A 95% interval on an accuracy p from n examples is about</p><div class="eq">±1.96·√(p(1 − p)/n)</div><div class="eq hook">Small slices need uncertainty, not just point estimates.</div>' },
        { t: 'num', q: 'A critical slice has 200 examples and 90% accuracy. About how many percentage points either side is a 95% interval?', a: 4.16, tol: 0.3, ex: '±1.96 × √(0.9 × 0.1 / 200) ≈ ±4.2 points. Quadruple the examples to halve it.', gen: gen.ciHalf },
        { t: 'mc', q: 'On 200 slice examples the canary scores 2 points below production, around 90% accuracy. Production’s number comes from far more traffic. The canary’s 95% interval is about ±4.2 points. What can you conclude?', o: ['It could easily be noise: get more examples before concluding (or block, if the slice is safety-critical)', 'It’s unlikely to be noise: treat it as a real regression, after checking for duplicates and representativeness', 'The canary is actually better', 'Intervals don’t apply to slices'], ex: '2 points is well inside ±4.2: a sample this size moves that much by chance.', gen: gen.noise },
        { t: 'mc', q: '60 of the 200 slice examples are near-duplicates of one conversation. What does that do?', o: ['Shrinks the effective sample, so the real uncertainty is larger than it looks', 'Nothing, as long as the labels are right', 'Makes the estimate more precise', 'Only affects latency'], ex: 'Duplicates aren’t independent evidence. The interval formula assumes independent examples, so it now understates the uncertainty.' },
        { t: 'num', q: 'A slice runs at about 90% accuracy. Roughly how many examples give a 95% interval of ±2 points?', a: 865, tol: 17, ex: 'n = 1.96² × p(1 − p) / h² = 3.84 × 0.09 / 0.02² ≈ 865.', gen: gen.sampleSize },
        { t: 'mc', q: 'On 200 slice examples the canary scores 8 points below production, around 90% accuracy. Production’s number comes from far more traffic. The canary’s 95% interval is about ±4.2 points. What can you conclude?', o: ['It’s unlikely to be noise: treat it as a real regression, after checking for duplicates and representativeness', 'It could easily be noise: get more examples before concluding (or block, if the slice is safety-critical)', 'The canary is actually better', 'Intervals don’t apply to slices'], ex: '8 points is well outside ±4.2, so chance alone is an unlikely explanation.', gen: gen.noise },
        { t: 'mc', q: 'A safety-critical slice has too few canary examples to tell whether it regressed. What should the promotion system do?', o: ['Hold or block until there’s enough evidence: uncertainty itself can block', 'Promote, since no regression was shown', 'Average it into the overall metric', 'Drop the slice'], ex: '“No evidence of a regression” isn’t evidence of no regression. For safety-critical slices, contracts can require a minimum count and a confidence bound.' },
      ],
      cards: [
        { id: 'ops-small-sample', front: 'A small slice looks worse. What do you need?', back: 'Its uncertainty, not just the point estimate.' },
        { id: 'ops-sample-checks', front: 'Why do duplicates matter in a small slice?', back: 'They make the sample look bigger than it is.' },
        { id: 'ops-uncertainty-blocks', front: 'Can uncertainty alone block a rollout?', back: 'Yes, for a safety-critical slice.' },
        { id: 'ops-contract-stats', front: 'More examples make a slice’s interval wider or narrower?', back: 'Narrower.' },
      ],
    },
    {
      id: 'ops-5',
      title: 'Leakage vs training-serving skew',
      steps: [
        { t: 'learn', title: 'Two different failures', body: '<p><b>Data leakage:</b> training or evaluation used information that wouldn’t be available at prediction time. Predicting cancellation tomorrow from “canceled within the next 7 days” is leakage.</p><div class="eq hook">Leakage = the model saw information it shouldn’t have.</div><p><b>Training-serving skew:</b> a feature or preprocessing step is computed differently in training and production. Training counts transactions in the 24 hours before prediction; production counts the previous calendar day. Same intended feature, different meaning.</p><div class="eq hook">Skew = the model sees something different in production.</div>' },
        { t: 'mc', q: 'A churn model uses “customer canceled within the next 7 days” to predict cancellation tomorrow. What is this?', o: ['Data leakage', 'Training-serving skew', 'Neither: this is fine'], ex: 'The model used information it couldn’t legitimately have at prediction time, so its offline numbers are too good to be true.', gen: gen.leakOrSkew },
        { t: 'mc', q: 'Training normalizes with fixed training-set mean and std; serving recomputes them from each batch. What is this?', o: ['Training-serving skew', 'Data leakage', 'Neither: this is fine'], ex: 'The feature means something different in production than it did in training: the model sees inputs unlike the ones it learned from.', gen: gen.leakOrSkew },
        { t: 'mc', q: 'Serving uses a newer tokenizer version than training did. What is this?', o: ['Training-serving skew', 'Data leakage', 'Neither: this is fine'], ex: 'The model sees different token ids in production than it learned from.', gen: gen.leakOrSkew },
        { t: 'learn', title: 'Preventing skew', body: '<p>Documentation alone doesn’t prevent skew. Stronger protections:</p><ul><li>version the preprocessing code and tie its artifacts to the model version</li><li>reuse the same transformation implementation where possible</li><li>test schemas, feature order, missing values, tokenization and normalization</li><li>run offline/online <b>parity tests</b>: push the same raw examples through both pipelines and check the model inputs match</li></ul><div class="eq hook">Same raw input → same model feature.</div>' },
        { t: 'mc', q: 'What is the strongest test against training-serving skew?', o: ['Run the same raw examples through both pipelines and check the model inputs match', 'Write detailed documentation of each feature', 'Compare training and validation loss', 'Retrain more often'], ex: 'A parity test checks the thing that matters directly: identical raw inputs must become identical features.' },
        { t: 'tf', q: 'Clear documentation of the preprocessing is enough to prevent training-serving skew.', a: false, ex: 'Docs drift from code. Version the preprocessing with the model, share the implementation, and test parity.' },
        { t: 'match', q: 'Match each term to its meaning.', p: [['Data leakage', 'Used information unavailable at prediction time'], ['Training-serving skew', 'Feature computed differently in production'], ['Parity test', 'Same raw input → same model feature'], ['Versioned preprocessing', 'Tied to the model version']], ex: 'Leakage inflates offline results; skew breaks the model quietly in production.' },
      ],
      cards: [
        { id: 'ops-leakage', front: 'What is data leakage?', back: 'Training used information not available at prediction time.' },
        { id: 'ops-skew', front: 'What is training-serving skew?', back: 'A feature computed differently in training and production.' },
        { id: 'ops-norm-skew', front: 'Training uses one normalization and serving another. What is it?', back: 'Training-serving skew.' },
        { id: 'ops-future-leak', front: 'Training uses information from after the prediction time. What is it?', back: 'Data leakage.' },
        { id: 'ops-parity', front: 'What is a parity test?', back: 'Same raw input through both pipelines, checking the features match.' },
      ],
    },
    {
      id: 'ops-6',
      title: 'Safe rollout and rollback',
      steps: [
        { t: 'learn', title: 'Not straight to 100%', body: '<div class="eq hook">offline gates → shadow → canary → gradual ramp → full production</div><p><b>Shadow:</b> the candidate gets real production inputs but doesn’t affect users. Good for real-traffic validation, latency, resources, output comparison and slices.</p><p><b>Canary:</b> a small share of users get the candidate (5% → 10% → 25% → 50% → 100%, paced by risk and traffic). Watch <b>system</b> metrics (latency, throughput, errors, OOMs, memory/CPU/GPU) and <b>model/product</b> metrics (quality, critical slices, KPIs, guardrails, complaints, calibration).</p>' },
        { t: 'mc', q: 'Every offline gate passes. What comes next?', o: ['Shadow deployment on real traffic', 'Full production immediately', 'A canary at 50%', 'Retraining'], ex: 'Offline data isn’t production. Shadow first, then a small canary, then a gradual ramp.' },
        { t: 'mc', q: '“The candidate gets real production inputs, but users never see its outputs.” Which stage is this?', o: ['Shadow deployment', 'Offline gates', 'Canary', 'Gradual ramp'], ex: 'The path is offline gates → shadow → canary → gradual ramp → full production, with rollback ready at every step.', gen: gen.stage },
        { t: 'mc', q: 'During a canary you watch p99 latency. What kind of signal is it?', o: ['A system metric', 'A model or product metric', 'Neither: not worth watching'], ex: 'System metrics ask whether the service is healthy: latency, throughput, errors, OOMs, resources.', gen: gen.metricKind },
        { t: 'mc', q: 'During a canary you watch recall on the self-harm slice. What kind of signal is it?', o: ['A model or product metric', 'A system metric', 'Neither: not worth watching'], ex: 'Model and product metrics ask whether it behaves: quality, critical slices, KPIs, guardrails, complaints, calibration.', gen: gen.metricKind },
        { t: 'learn', title: 'Rollback is part of the design', body: '<p>Don’t improvise a rollback after something breaks. Before rollout, decide:</p><ul><li>which metrics trigger it, and who or what fires it</li><li>how fast it happens</li><li>which <b>immutable</b> model version, and which serving and preprocessing configuration, come back</li></ul><p>Keep the previous production model immediately available.</p><div class="eq hook">Promotion without rollback is incomplete deployment design.</div>' },
        { t: 'mc', q: 'Which of these must be decided before the rollout starts?', o: ['What triggers a rollback, who fires it, and which immutable version it restores', 'Who to blame if it fails', 'Whether to keep the old model around', 'Nothing: decide when something breaks'], ex: 'A rollback plan made during an incident is slow and error-prone. Make it part of the contract.' },
        { t: 'tf', q: 'Passing every offline gate means the model can go straight to 100% of traffic.', a: false, ex: 'Offline gates can’t see real traffic, real load or real users. Shadow, canary and a gradual ramp catch what they miss.' },
        { t: 'mc', q: '“Traffic moves 10% → 25% → 50% as the monitored metrics hold.” Which stage is this?', o: ['Gradual ramp', 'Shadow deployment', 'Offline gates', 'Rollback'], ex: 'The path is offline gates → shadow → canary → gradual ramp → full production, with rollback ready at every step.', gen: gen.stage },
        { t: 'explain', kind: 'defend', secs: 90, q: 'How would you design CI/CD for an ML platform used by 50 researchers?', model: '<p>I’d make CI/CD enforce a <b>promotion contract</b> across four layers. First, <b>software and integration correctness</b>: API contracts, serialization, database writes, dependency compatibility and preprocessing parity. Second, <b>serving performance</b>: latency, throughput, memory footprint, startup and failure behavior. Third, <b>ML behavior</b>: primary metrics, critical slices, sanity and regression sets, with thresholds defined in advance so teams aren’t making ad hoc launch decisions. Fourth, <b>rollout</b>: shadow, then canary, then a gradual ramp with explicit monitoring and rollback criteria. The prior production version stays immutable and immediately available if either system or model guardrails regress.</p>', rubric: ['A promotion contract enforced by the pipeline, set in advance', 'Software/integration correctness, including preprocessing parity', 'Serving performance: latency, throughput, memory', 'ML behavior: metrics, critical slices, sanity sets', 'Shadow → canary → ramp with monitoring and an immutable rollback target'] },
      ],
      cards: [
        { id: 'ops-rollout', front: 'What comes right after offline gates?', back: 'A shadow deployment.' },
        { id: 'ops-shadow', front: 'What is a shadow deployment?', back: 'Real traffic, but users never see its outputs.' },
        { id: 'ops-canary-watch', front: 'What is a canary?', back: 'A small share of users get the new model first.' },
        { id: 'ops-rollback', front: 'What should a rollback restore?', back: 'The previous immutable model version.' },
        { id: 'ops-design', front: 'Is rollback part of deployment design?', back: 'Yes: promotion without rollback is incomplete.' },
      ],
    },
  ],
};
