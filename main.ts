// gta6-trophy-rarity-forecast-01: forecast every trophy's Exophase earn rate at the case's target.
//
// Exophase refreshes a game's counts on its own schedule, hours apart, while the house reads the page every ten
// minutes. So a case's history is mostly the same page repeated: a few reads that carry a refresh, and a long run of
// reads that repeat the previous one. Fitting a line through every read would be pulled to a slope of zero by the
// plateau, and would measure the reading clock rather than the game. The model here therefore works on snapshots.
//
// 1. Snapshots. Keep the first read and the last read, and drop every read whose rates are identical to the last
//    read kept: what is left is the reads that actually carried a change, plus the endpoints.
// 2. Trend. For each trophy, on the snapshots that carry its rate, fit logit(rate) against ln(1 + hours since the
//    first snapshot), over at most the last five snapshots, then damp the slope by 0.6 and cap it at +/-0.5. A new
//    game slows after its opening burst, so a raw slope carried days forward overshoots. The median is that fit read
//    at the target, from the last rate.
// 3. Spread. Calibrated from the jumps the case itself shows, not from a constant: for every pair of consecutive
//    snapshots and every trophy present in both, the logit change per sqrt day, and the upper quartile of their
//    absolute values is the case's volatility. Most trophies do not move between two refreshes -- their rate is
//    rounded to 0.01% and the population's growth does not shift it -- so the median of every jump is zero and says
//    nothing about the trophies that do move; the jumps of exactly zero are dropped before the quantile is read.
//    sigma = 1.25 * vol * sqrt(days left after the last read), so the 90% band runs
//    a little wider than a pure random walk, because who plays, and how far, keeps changing. A history that shows no
//    movement at all has no evidence of movement, so it gets a wide spread rather than a narrow one.
// 4. Edges. A trophy at 100% is already earned by every tracked player and cannot move. A trophy below the page's
//    0.01% resolution moves in steps of a few players, so its spread is capped.
//
// The five quantiles are the normal's quantiles of that spread around the median rate, rounded to 4 decimals and
// forced non-decreasing. Node 22, standard library only, no network, no build step, deterministic: the same input
// writes the same bytes. MAKEFUN_INPUT_DIR (default /input) is read for every *.in; MAKEFUN_OUTPUT_DIR (default
// /output) gets one *.out each. MAKEFUN_DEBUG=1 prints each trophy's fit to stdout and changes no output byte.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** How much of a fitted trend survives into the forecast. */
const DAMP = 0.6;
/** The steepest trend, in logit units per unit of ln(1 + hours), that a forecast will carry. */
const SLOPE_MAX = 0.5;
/** How many of the most recent snapshots a trend is fitted over. */
const WINDOW = 5;
/** The spread a case's own volatility is scaled by: a little wider than a pure random walk. */
const SIGMA_K = 1.25;
/** Which quantile of the observed jumps the spread is read at: most trophies sit still between two refreshes
 *  (their rate is rounded to 0.01%, and the population's growth does not move it), so the median of every jump is
 *  zero and says nothing. The upper quartile is the jump a trophy that does move actually makes. */
const VOL_QUANTILE = 0.75;
/** The spread a case with no observed movement gets, per sqrt day: a history that shows nothing has no evidence
 *  that nothing will happen -- Exophase refreshes on its own clock, and one refresh moves every rate at once. */
const FLAT_K = 2.70;
/** The fewest days a flat history's spread is measured over: a 90-minute practice case still needs a band wide
 *  enough to hold a refresh it has not seen. */
const FLAT_DAYS = 1.0;
/** The spread a case gets per sqrt day of horizon even when its own reads barely move: a tracked population that
 *  doubles while the earners do not moves every rate at once, and the score punishes a band that is too narrow far
 *  more than one that is too wide. */
const SIGMA_FLOOR_PER_DAY = 0.45;
/** The fewest logit units any row is given, whatever its horizon. */
const SIGMA_MIN = 0.20;
const SIGMA_MAX = 3.20;
/** A rate this high is every tracked player, so the trophy cannot move. */
const SATURATED = 99.5;
/** The spread a saturated trophy gets: it cannot move. */
const SATURATED_SIGMA = 0.02;
/** A rate this low is below the page's resolution; its earners are a handful of players. */
const RARE = 0.01;
/** The widest spread a trophy below the resolution is given. */
const RARE_SIGMA = 0.45;
/** A read with no usable clock or population cannot be placed on the timeline. */
const MIN_POPULATION = 1;

interface Read {
  slot?: string;
  at: string;
  players_tracked: number;
  trophies_earned?: number;
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

const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;

const clamp = (lo: number, hi: number, x: number): number => Math.min(hi, Math.max(lo, x));
/** A rate in percent, clamped the way the scorer clamps it before the logit. */
const clampRate = (x: number): number => clamp(0.01, 99.99, x);
/** logit of a rate in percent, at the scorer's clamping. */
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
/** A logit back to a rate in percent, clamped to what the page can print. */
const rate = (z: number): number => clampRate(100 / (1 + Math.exp(-z)));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

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
  const s = sxy / sxx;
  return Number.isFinite(s) ? s : null;
}

/** True when two reads carry the same rates: null for null, and equal numbers for equal numbers. */
function sameRates(a: Read['rates'], b: Read['rates']): boolean {
  if (a === b) return true;
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (x === y) continue;
    if (isNum(x) && isNum(y) && x === y) continue;
    return false;
  }
  return true;
}

/** The reads a case was built on, placed on the clock, oldest first. */
function reads(hist: unknown): Read[] {
  if (!Array.isArray(hist)) return [];
  const out: Read[] = [];
  for (const h of hist as Read[]) {
    if (!h || !Number.isFinite(Date.parse(h.at))) continue;
    if (!isNum(h.players_tracked) || h.players_tracked < MIN_POPULATION) continue;
    // A read that carries no rate vector at all carries nothing this forecast can use.
    if (!Array.isArray(h.rates)) continue;
    out.push(h);
  }
  return out;
}

/** The first read, the last read, and every read in between that carried a change: the plateaus dropped. */
function snapshots(hist: Read[]): Read[] {
  if (hist.length === 0) return [];
  const out: Read[] = [hist[0]];
  for (let k = 1; k < hist.length; k++) {
    if (!sameRates(hist[k].rates, out[out.length - 1].rates)) out.push(hist[k]);
  }
  const last = hist[hist.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

/** One trophy's rate on every snapshot that carried it, with its position on the ln(1 + hours) clock. */
interface Series {
  x: number[];
  y: number[];
  rate: number;
  n: number;
}

function seriesOf(snaps: Read[], index: number, t0: number): Series {
  const x: number[] = [];
  const y: number[] = [];
  let last = Number.NaN;
  for (const s of snaps) {
    const r = s.rates?.[index];
    if (!isNum(r)) continue;
    x.push(Math.log1p(Math.max(0, (Date.parse(s.at) - t0) / HOUR_MS)));
    y.push(logit(r));
    last = r;
  }
  return { x, y, rate: last, n: x.length };
}

/** The case's volatility: the jump per sqrt day that a trophy which moves actually makes, over every jump that is not
 *  zero. See VOL_QUANTILE for why the zero jumps are dropped and why the quantile is the upper quartile. */
function volatility(snaps: Read[]): number {
  const vs: number[] = [];
  for (let k = 1; k < snaps.length; k++) {
    const dtDays = (Date.parse(snaps[k].at) - Date.parse(snaps[k - 1].at)) / DAY_MS;
    if (!(dtDays > 0)) continue;
    const a = snaps[k - 1].rates;
    const b = snaps[k].rates;
    if (!Array.isArray(a) || !Array.isArray(b)) continue;
    const n = Math.min(a.length, b.length);
    const root = Math.sqrt(dtDays);
    for (let i = 0; i < n; i++) {
      const ra = a[i];
      const rb = b[i];
      if (!isNum(ra) || !isNum(rb)) continue;
      const v = Math.abs(logit(rb) - logit(ra)) / root;
      // A jump of exactly zero is a trophy that did not move, not a trophy that moves slowly.
      if (v > 1e-9) vs.push(v);
    }
  }
  if (vs.length === 0) return 0;
  vs.sort((x, y) => x - y);
  return vs[Math.min(vs.length - 1, Math.floor(VOL_QUANTILE * (vs.length - 1)))];
}

function forecast(c: Case): Array<{ id: number; q: number[] }> {
  const hist = reads(c.history);
  const ids = c.trophies.map((t) => t.id);
  const snaps = snapshots(hist);

  // A case with no usable read still has to write a valid row: place it on the checkpoint and let the horizon carry
  // the uncertainty.
  const atMs = Number.isFinite(Date.parse(c.at)) ? Date.parse(c.at) : 0;
  const t0 = snaps.length > 0 ? Date.parse(snaps[0].at) : atMs;
  const tLast = snaps.length > 0 ? Date.parse(snaps[snaps.length - 1].at) : atMs;
  const targetMs = Number.isFinite(Date.parse(c.target)) ? Date.parse(c.target) : tLast;
  const daysLeft = Math.max(0, (targetMs - tLast) / DAY_MS);

  // The spread this case's own history shows. A history with no jump at all is not a case with no uncertainty: it
  // is a case with no evidence, so it gets the wide fallback instead of the narrow one.
  const vol = volatility(snaps);
  const floor = Math.max(SIGMA_MIN, SIGMA_FLOOR_PER_DAY * Math.sqrt(daysLeft));
  const sigma =
    vol > 0
      ? clamp(floor, SIGMA_MAX, SIGMA_K * vol * Math.sqrt(daysLeft))
      : clamp(SIGMA_MIN, SIGMA_MAX, FLAT_K * Math.sqrt(Math.max(daysLeft, FLAT_DAYS)));

  const xTarget = snaps.length > 0 ? Math.log1p(Math.max(0, (targetMs - t0) / HOUR_MS)) : 0;
  const debug = process.env.MAKEFUN_DEBUG === '1';
  if (debug) {
    console.log(`# ${c.id}: reads=${hist.length} snaps=${snaps.length} vol=${vol.toFixed(6)} sigma=${sigma.toFixed(6)} daysLeft=${daysLeft.toFixed(4)}`);
  }

  return ids.map((id, i) => {
    const s = seriesOf(snaps, i, t0);
    let m: number;
    let sig: number;

    if (s.n === 0) {
      // No read carried this trophy: nothing to say about it but that it is rare.
      m = logit(RARE);
      sig = Math.min(sigma, RARE_SIGMA);
    } else if (s.rate >= SATURATED) {
      // Earned by every tracked player: it has nowhere to go.
      m = logit(99.99);
      sig = SATURATED_SIGMA;
    } else if (s.rate <= RARE) {
      // Below the page's resolution: a few players, so it can tick up but cannot run.
      m = logit(RARE);
      sig = Math.min(sigma, RARE_SIGMA);
    } else if (s.n < 2) {
      // One snapshot is not a trend. Keep the rate where it is.
      m = logit(s.rate);
      sig = sigma;
    } else {
      const from = Math.max(0, s.n - WINDOW);
      const xs = s.x.slice(from);
      const ys = s.y.slice(from);
      const raw = slope(xs, ys) ?? 0;
      const damped = clamp(-SLOPE_MAX, SLOPE_MAX, raw * DAMP);
      m = logit(s.rate) + damped * (xTarget - s.x[s.n - 1]);
      sig = sigma;
      if (debug) {
        console.log(
          `#   id=${id} snaps=${s.n} win=${xs.length} slope=${raw.toFixed(6)} damped=${damped.toFixed(6)} m=${m.toFixed(6)} sigma=${sig.toFixed(6)} last=${s.rate}`,
        );
      }
    }
    if (debug && s.n < 2) {
      console.log(`#   id=${id} snaps=${s.n} edge m=${m.toFixed(6)} sigma=${sig.toFixed(6)} last=${s.rate}`);
    }

    const q: number[] = [];
    for (const z of Z) {
      const v = round4(rate(m + z * sig));
      q.push(q.length > 0 && v < q[q.length - 1] ? q[q.length - 1] : v);
    }
    return { id, q };
  });
}

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const files = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of files) {
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const rows = forecast(c);
  const id = typeof c.id === 'string' && c.id.length > 0 ? c.id : file.slice(0, -'.in'.length);
  // One row per line: readable, and the bytes stay a pure function of the case.
  const body = rows.map((r) => `{"id":${r.id},"q":[${r.q.map((q) => q.toFixed(4)).join(',')}]}`).join(',\n');
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `{"trophies":[\n${body}\n]}\n`, 'utf8');
}
