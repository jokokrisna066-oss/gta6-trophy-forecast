# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jokokrisna066-oss/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jokokrisna066-oss/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's Exophase earn rate at a target instant, as five
quantiles on the logit scale, for make.fun's
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty, $250 in USDC.

Exophase prints an earn rate as a percentage of the players it tracks, so the number of players who
earned a trophy is `E = rate × players_tracked / 100` — a small integer that moves smoothly, while
the printed rate falls whenever the tracked population grows. This entry therefore fits `E` and the
tracked population `N` separately, both against `t = ln(1 + hours since the first read)`, and puts
them back together at the target: `rate = 100 × E/N`.

Each trophy's own slope is trusted in proportion to how many earners back it, so a trophy with 3,600
earners keeps its own fit while a trophy with 3 earners is pulled toward what the trophies as a group
do — a count that small is rounding noise. Earners never fall: a trophy already earned stays earned.
The spread is on the logit scale, where the scorer measures its miss, and combines the counting noise
of `E`, the uncertainty a growing population adds to the ratio, and a time term for what the history
cannot see.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, no build step, deterministic: the same input writes the
same bytes. It reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out`
per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

---

<sub>Entered on [make.fun](https://make.fun/?ref=8jdc4key&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
