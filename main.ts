// gta6-trophy-rarity-forecast-01: forecast every trophy's Exophase earn rate at the case's target.
//
// Exophase prints an earn rate as a percent of the players it tracks, so the count of players who earned a trophy is
// E = rate * players_tracked / 100 -- a small integer that moves smoothly, while the rate itself falls whenever the
// tracked population grows. So the fit is done on E and on the tracked population N separately, both against
// t = ln(1 + hours since the first read), and the two are put back together at the target:
//
//   nsl  = slope of ln(N) on t, clamped to [0, 1.5]          (N never shrinks; a refresh can triple it in one step)
//   Np   = N_last * exp(nsl * (tT - t_last)), never below N_last
//   sl_i = slope of ln(E_i) on t for trophy i, shrunk toward the median slope over all trophies (weight 2.0), and
//          clamped to [-0.5, 2.5]
//   Ep_i = E_i,last * exp(sl_i * (tT - t_last)), never below 0.3   (a tenth of one earner at 30k players)
//   r_i  = 100 * Ep_i / Np, clamped to [0.01, 99.99]
//
// The spread is on the logit scale, where the scorer measures its miss, and has three parts: the counting noise of E
// (1/sqrt(E)), the uncertainty a growing population adds to the ratio (proportional to how far ln(N) is extrapolated),
// and a time term (0.22 per sqrt day of horizon left after the last read) for what the history cannot see -- a
// refresh, a patch, a week of players who stop playing. The five quantiles are the normal's quantiles of that spread
// around the median rate, rounded to 4 decimals and forced non-decreasing.
//
// Node 22, standard library only, no network, no build step, deterministic: the same input writes the same bytes.
// MAKEFUN_INPUT_DIR (default /input) is read for every *.in; MAKEFUN_OUTPUT_DIR (default /output) gets one *.out each.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** How much the median slope over all trophies outweighs one trophy's own fit. */
const SHRINK_WEIGHT = 2.0;
/** Spread, in logit units, per sqrt day of horizon the history does not cover. */
const SIGMA_TIME = 0.22;
/** Spread the tracked population's growth adds, per unit of extrapolated ln(N) and per unit of nsl. */
const SIGMA_N = 0.10 * 0.5;
/** The widest spread any row is given, in logit units. */
const SIGMA_MAX = 2.2;
/** The smallest count of earners a forecast will entertain. */
const E_FLOOR = 0.3;
/** A rate this high means every tracked player has the trophy, so its earners track the population. */
const SATURATED = 99.5;
/** The horizon, in hours, below which a case is treated as having no history to fit. */
const MIN_FIT_READS = 2;

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Array<{ id: number }>;
  history: Read[];
}

const clamp = (lo: number, hi: number, x: number): number => Math.min(hi, Math.max(lo, x));
/** A rate in percent, clamped the way the scorer clamps it before the logit. */
const clampRate = (x: number): number => clamp(0.01, 99.99, x);
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const HOUR_MS = 3_600_000;
const hoursBetween = (fromMs: number, toMs: number): number => (toMs - fromMs) / HOUR_MS;
const finite = (x: number): number => (Number.isFinite(x) ? x : 0);
const ms = (iso: string, fallback: number): number => {
  const v = Date.parse(iso);
  return Number.isFinite(v) ? v : fallback;
};

/** Least-squares slope of ys on xs; null when there is nothing to fit (fewer than two points, no spread in xs). */
function slope(xs: number[], ys: number[]): number | null {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (!(sxx > 1e-12)) return null;
  return finite(sxy / sxx);
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** One trophy's five quantiles, from a rate and a logit spread, rounded and forced non-decreasing. */
function quantiles(rate: number, sigma: number): number[] {
  const m = logit(rate);
  const q: number[] = [];
  for (const z of Z) {
    const v = round4(clamp(0, 100, 100 / (1 + Math.exp(-(m + z * sigma)))));
    q.push(q.length > 0 && v < q[q.length - 1] ? q[q.length - 1] : v);
  }
  return q;
}

/** The trophy's rate in the last read that carried it; null when no read did. */
function lastRate(hist: Read[], i: number): number | null {
  for (let k = hist.length - 1; k >= 0; k--) {
    const r = hist[k]?.rates?.[i];
    if (typeof r === 'number' && Number.isFinite(r)) return r;
  }
  return null;
}

function forecast(c: Case): Array<{ id: number; q: number[] }> {
  const hist = (Array.isArray(c.history) ? c.history : []).filter(
    (h) => h && Number.isFinite(h.players_tracked) && Number.isFinite(Date.parse(h.at)),
  );
  const ids = c.trophies.map((t) => t.id);

  // Nothing to fit: keep each trophy's last rate, and widen only with the horizon the case leaves open.
  if (hist.length < MIN_FIT_READS) {
    const t0 = hist.length > 0 ? Date.parse(hist[0].at) : ms(c.at, 0);
    const tL = hist.length > 0 ? Date.parse(hist[hist.length - 1].at) : t0;
    const gap = Math.max(0, hoursBetween(tL, ms(c.target, tL)));
    const sigma = Math.min(SIGMA_MAX, SIGMA_TIME * Math.sqrt(gap / 24));
    return ids.map((id, i) => ({ id, q: quantiles(lastRate(hist, i) ?? 0.01, sigma) }));
  }

  const t0 = Date.parse(hist[0].at);
  const lastMs = Date.parse(hist[hist.length - 1].at);
  const targetMs = ms(c.target, lastMs);
  const ts = hist.map((h) => Math.log1p(Math.max(0, hoursBetween(t0, Date.parse(h.at)))));
  const tLast = ts[ts.length - 1];
  const tT = Math.log1p(Math.max(0, hoursBetween(t0, targetMs)));
  const gapT = Math.max(0, tT - tLast);
  const gapH = Math.max(0, hoursBetween(lastMs, targetMs));

  // The tracked population: ln(N) on t, never falling, never trusted to grow faster than e^1.5 per unit of t.
  const lnN = hist.map((h) => Math.log(Math.max(1, h.players_tracked)));
  const nsl = clamp(0, 1.5, slope(ts, lnN) ?? 0);
  const nLast = Math.max(1, hist[hist.length - 1].players_tracked);
  const nPred = Math.max(nLast, nLast * Math.exp(nsl * gapT));

  // Each trophy's count of earners on every read that carried it, and the slope of its ln(E) on t.
  const eLast: number[] = new Array(ids.length).fill(0);
  const own: Array<number | null> = new Array(ids.length).fill(null);
  const fitted: number[] = [];
  for (let i = 0; i < ids.length; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let k = 0; k < hist.length; k++) {
      const r = hist[k].rates?.[i];
      if (typeof r !== 'number' || !Number.isFinite(r)) continue;
      const earners = (Math.max(0.01, r) * Math.max(1, hist[k].players_tracked)) / 100;
      xs.push(ts[k]);
      ys.push(Math.log(earners));
      eLast[i] = earners;
    }
    const s = slope(xs, ys);
    own[i] = s;
    if (s !== null) fitted.push(s);
  }
  const medianSlope = median(fitted);

  const sigH = SIGMA_TIME * Math.sqrt(gapH / 24);

  return ids.map((id, i) => {
    // A saturated trophy (every tracked player has it) can only move with N: its earners are the tracked players,
    // so its slope is the population's, and the shrink toward other trophies would be plainly wrong there.
    const saturated = (lastRate(hist, i) ?? 0) >= SATURATED;
    // Shrink a rare trophy's own fit toward what the trophies as a group do; the median weighs SHRINK_WEIGHT.
    const shrunk = saturated ? nsl
      : own[i] === null ? medianSlope
      : (own[i]! + SHRINK_WEIGHT * medianSlope) / (1 + SHRINK_WEIGHT);
    const sl = clamp(-0.5, 2.5, finite(shrunk));

    // Earners are never lost: a trophy already earned stays earned, so E cannot fall below its last count.
    const ePred = Math.min(nPred, Math.max(eLast[i], eLast[i] * Math.exp(sl * gapT), E_FLOOR));
    const rate = clampRate((100 * ePred) / nPred);

    const sigE = 1 / Math.sqrt(Math.max(ePred, 1));
    const sigN = SIGMA_N * Math.abs(nsl) * gapT;
    const sigma = Math.min(SIGMA_MAX, Math.sqrt(sigE * sigE + sigN * sigN + sigH * sigH));

    return { id, q: quantiles(rate, sigma) };
  });
}

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const files = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of files) {
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!Array.isArray(c.levels) || c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const rows = forecast(c);
  const id = typeof c.id === 'string' && c.id.length > 0 ? c.id : file.slice(0, -'.in'.length);
  // One row per line: readable, and the bytes stay a pure function of the case.
  const body = rows.map((r) => `{"id":${r.id},"q":[${r.q.map((q) => q.toFixed(4)).join(',')}]}`).join(',\n');
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `{"trophies":[\n${body}\n]}\n`, 'utf8');
}
