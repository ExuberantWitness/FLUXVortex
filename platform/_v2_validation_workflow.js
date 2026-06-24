export const meta = {
  name: 'robowing-validation-analysis',
  description: 'Multi-perspective + adversarial analysis of strip-LDVM vs RoboEagle data.md (530 pts)',
  phases: [
    { title: 'Regime' },
    { title: 'Adversarial' },
    { title: 'Synthesis' },
  ],
}

const JSON_PATH = '/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/platform/_v2_validation.json'
const DIR = '/home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV/platform'

const MODEL_CONTEXT = `
MODEL UNDER TEST: strip-theory LDVM for the RoboEagle flapping wing (Drones 2025).
Each spanwise strip = a 2D unsteady LDVM (discrete-vortex). Per strip:
 - LIFT = instantaneous pressure normal force lift_p (validated: steady CL=2*pi*alpha; reliable cycle-mean under large plunge).
 - THRUST = leading-edge suction F_s = pi*rho*U^2*c*A0^2 (Garrick/Ramesh-2020), validated to 92% of analytic Garrick heave thrust.
 - airfoil = real NACA-2406 camber (m=2% at p=40%) -> gives the 0deg-AoA lift offset.
 - LE suction SATURATES at LESP_crit=0.20 (separation cap); NO discrete LEV shedding (lev_shed=False, which diverges at +-45deg flap).
 - 3D-downwash correction = CONSTANT x AR/(AR+2)=0.74 (AR=5.6).
 - profile drag cd0=0 in this sweep (OFF). Kinematics: flap +-45deg, plunge=y*thetadot, twist phase=90deg.
KNOWN, DECLARED LIMITATIONS (do NOT re-discover as if novel; judge their SEVERITY against the data):
 (a) 2D strip has NO induced drag -> THRUST is gross-suction only, will read more positive than the net-DRAG data.
 (b) the 0.74 3D correction is CONSTANT -> cannot capture an AoA-dependent stall.
 (c) NO dynamic-stall model -> high-AoA lift expected to over-predict.
 (d) twist handling is suspect (earlier: twist reduced model lift, but data shows lift RISES with twist to ~15-22deg then drops).
DATA: RoboEagle measured lift & thrust (grams-force -> N) vs twist / frequency / wind speed / body AoA.
Results JSON at ${JSON_PATH}: rows = [{fig, kind(L/T), sweep, wind, aoa, freq, twist, x, meas_N, model_N}].
Per-regime stats: run \`cd ${DIR} && python _v2_validate_all.py --stats\`.
`

const REGIME_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'n_points', 'trend_verdict', 'magnitude_verdict', 'numbers', 'physical_reading', 'severity'],
  properties: {
    dimension: { type: 'string' },
    n_points: { type: 'integer' },
    trend_verdict: { type: 'string', enum: ['matches', 'partly', 'wrong'], description: 'does the model follow the data trend along the swept variable?' },
    magnitude_verdict: { type: 'string', enum: ['within_10pct', 'within_25pct', 'off_25_50pct', 'off_gt_50pct', 'mixed'] },
    numbers: { type: 'string', description: 'concrete per-condition ratios/correlations/offsets, e.g. "8m/s lift ratio 1.11; trend corr 0.99; 6m/s flat vs model rising"' },
    physical_reading: { type: 'string', description: 'what the agreement/gap means physically' },
    severity: { type: 'string', enum: ['ok', 'minor', 'major'], description: 'severity of the gap for the 5%-calibration goal' },
  },
}

const ADV_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'strongest_failures', 'curvefit_risk', 'overstated_claims'],
  properties: {
    verdict: { type: 'string', enum: ['agreement_is_real', 'partly_real', 'largely_curvefit_or_trivial'] },
    strongest_failures: { type: 'array', items: { type: 'string' }, description: 'the 2-4 most damaging concrete failures vs the data' },
    curvefit_risk: { type: 'string', description: 'are free params (0.74, lesp_crit, camber, cd0) tuned to the data, or independent? where is the risk?' },
    overstated_claims: { type: 'array', items: { type: 'string' }, description: 'any claim that the model "matches" which the data does not support' },
  },
}

phase('Regime')
const DIMENSIONS = [
  { key: 'lift-vs-frequency', prompt: 'Assess LIFT vs FREQUENCY: Fig18b (wind 6/8/10 m/s, 5deg, twist0) and Fig19b (AoA 0/5/10/15deg, 8m/s, twist0). The data shows lift RISES with frequency at 8/10 m/s and at 5-15deg AoA, but is FLAT at 6 m/s and at 0deg AoA. Does the model reproduce this rise AND its flat exceptions? Give per-condition magnitude ratios and trend correlations.' },
  { key: 'lift-vs-windspeed', prompt: 'Assess LIFT vs WIND SPEED: Fig18b at 6/8/10 m/s (5deg, twist0, vs frequency). Does model lift scale with wind speed like the data (6m/s~390g flat, 8m/s 626-794g, 10m/s 881-1159g)? Is the U-scaling right? Magnitude ratios per wind speed.' },
  { key: 'lift-vs-aoa', prompt: 'Assess LIFT vs BODY AoA: Fig19b at 0/5/10/15deg (8m/s, twist0). Data lift slope FLATTENS at high AoA (0->2.9N, 5->7.8, 10->12.1->stall). Does the model over-predict at high AoA (the missing dynamic stall)? Quantify the ratio growth with AoA.' },
  { key: 'lift-vs-twist', prompt: 'Assess LIFT vs TWIST amplitude: Fig17b (freq 1.4-2.6Hz), Fig18d (wind x freq), Fig19d (AoA). Data lift RISES with twist to ~15-22deg then DROPS. Does the model reproduce this non-monotonic twist response, or does it get the twist effect wrong (sign/shape)? This is a suspected bug.' },
  { key: 'thrust-all', prompt: 'Assess THRUST: Fig18a/19a (vs freq), Fig17a/18c/19c (vs twist). Data NET thrust is mostly DRAG (negative), becoming less negative at higher freq / higher wind / lower AoA. The 2D model has NO induced drag so its thrust reads too positive. Question: does the model at least capture the thrust TRENDS (direction vs freq/wind/AoA/twist)? Estimate the ~constant induced-drag offset that would reconcile model and data.' },
]
const regime = await pipeline(
  DIMENSIONS,
  d => agent(`${MODEL_CONTEXT}\n\nYOUR TASK (${d.key}): ${d.prompt}\n\nRead the JSON / run --stats, filter to your rows, compute concrete numbers. Be quantitative and HONEST about gaps.`,
    { label: d.key, phase: 'Regime', schema: REGIME_SCHEMA }),
)

phase('Adversarial')
const regimeSummary = JSON.stringify(regime.filter(Boolean), null, 1)
const adv = await parallel([1, 2, 3].map(i => () =>
  agent(`${MODEL_CONTEXT}\n\nThe per-regime analysis returned:\n${regimeSummary}\n\nYOU ARE ADVERSARIAL REVIEWER #${i}. Your job is to REFUTE the claim "this strip-LDVM is a validated RoboEagle aero model." Read the raw JSON yourself (do not trust the summary blindly). Find the most damaging concrete failures vs the data. Scrutinize whether the agreement is REAL physics or curve-fit/trivial: is the 0.74 correction, lesp_crit, camber, or cd0 secretly tuned to the data? Is the lift agreement trivial (lift just tracks the imposed AoA)? Where does the model contradict the data? Default to skepticism; cite specific figures/numbers.`,
    { label: `skeptic-${i}`, phase: 'Adversarial', schema: ADV_SCHEMA })))

phase('Synthesis')
const synth = await agent(
  `${MODEL_CONTEXT}\n\nPER-REGIME ANALYSIS:\n${regimeSummary}\n\nADVERSARIAL REVIEWS:\n${JSON.stringify(adv.filter(Boolean), null, 1)}\n\n` +
  `Write a COMPREHENSIVE, HONEST validation report (markdown) of the strip-LDVM vs the full RoboEagle data.md battery (530 points). Include: (1) a per-regime accuracy table (lift vs freq / wind / AoA / twist; thrust trends) with concrete ratios/correlations; (2) what is GENUINELY validated vs what is not; (3) the systematic gaps ranked by severity (dynamic stall, induced drag, twist) and the physics fix for each; (4) an explicit verdict on curve-fit risk; (5) the honest bottom line on how far from the 5%-calibration goal, and the highest-value next step. Do not overstate. Use the adversarial reviews to temper any optimism.`,
  { label: 'synthesis', phase: 'Synthesis' })

return { regime: regime.filter(Boolean), adversarial: adv.filter(Boolean), report: synth }
