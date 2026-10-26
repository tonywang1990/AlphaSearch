"""Choosing F from the trained bank (DOTA ADR 0452's Frontier): a greedy fill, then
best-improvement swaps gated at a paired val t, from each given start; crossval and an
ensemble search as the agent's honest estimates. Members are indices into the val `Stats`.

The selection objective is the IC from moments by default (SoM ADR 0008 selects on IC); an
objective with no moment form (ev, evf) is scored by infra in batches (`Infra.score_many`)."""
from dataclasses import dataclass

import numpy as np

from alphasearch.ensemble.moments import Stats, fit_weights, paired, paired_daily, score

SWAP_T = 2.0      # LU: unmargined swaps chosen on half the val days lost 0.06 on the other half


@dataclass(frozen=True)
class Cfg:
    k_max: int
    host: str | None = None
    corrector_w: float = 0.5
    swap_t: float = SWAP_T
    objective: str = "ic"


class Selector:
    def __init__(self, val: Stats, fit: Stats, cfg: Cfg, infra=None):
        self.val, self.cfg, self.infra = val, cfg, infra
        self.names = val.names
        cf, Sf = fit.moments(fit.ones())
        order = [fit.idx[n] for n in self.names]
        self.c_fit, self.S_fit = cf[order], Sf[np.ix_(order, order)]
        self._cache = {}

    # -- blends ---------------------------------------------------------------------------
    def weights(self, jj):
        """The host's fixed weights in host mode, else w = S^-1 c on the fit block."""
        if self.cfg.host:
            return np.array([1.0 if self.names[j] == self.cfg.host else self.cfg.corrector_w for j in jj])
        return fit_weights(self.c_fit[jj], self.S_fit[np.ix_(jj, jj)])

    def blend(self, jj):
        jj = list(jj)
        return jj, self.weights(jj)

    def legs(self, jj):
        return [(self.names[j], float(w)) for j, w in zip(*self.blend(jj), strict=True)]

    def ix(self, names):
        return [self.val.idx[n] for n in names if n in self.val.idx]

    # -- the objective --------------------------------------------------------------------
    def scorer(self, counts=None):
        """f(jj) -> the selection score on the val days (`counts` weights them: IC only), with
        `f.many(sets)`, one infra job for a whole batch on objectives with no moment form."""
        if self.cfg.objective == "ic":
            per = self.val.per_period(self.val.ones() if counts is None else counts)
            f = lambda jj: score(per, *self.blend(jj)) if len(jj) else float("nan")
            f.many = lambda sets: [f(s) for s in sets]
            return f
        assert counts is None, f"day weights need the moment form; {self.cfg.objective} has none"

        def many(sets):
            todo = [s for s in sets if frozenset(s) not in self._cache]
            for s, r in zip(todo, self.infra.score_many([self.legs(s) for s in todo], self.cfg.objective)
                            if todo else [], strict=True):
                self._cache[frozenset(s)] = r
            return [self._cache[frozenset(s)].value for s in sets]
        f = lambda jj: many([list(jj)])[0]
        f.many = many
        return f

    def t(self, counts=None):
        """t(a, b): the paired val t of member set a over b."""
        if self.cfg.objective == "ic":
            ones = self.val.ones() if counts is None else counts

            def t(a, b):
                g, se = paired(self.val, ones, self.blend(a), self.blend(b))
                return g / se if se > 0 else 0.0
            return t
        f, per = self.scorer(), dict(zip(map(str, self.val.days), self.val.period, strict=True))

        def t(a, b):
            f.many([list(a), list(b)])
            g, se = paired_daily(self._cache[frozenset(a)], self._cache[frozenset(b)], per)
            return g / se if se else 0.0
        return t

    def gain(self, new, old):
        """(gain, SE) of member set `new` over `old` on the val days; (0, 0) for the same set."""
        if set(new) == set(old):
            return 0.0, 0.0
        if self.cfg.objective == "ic":
            return paired(self.val, self.val.ones(), self.blend(new), self.blend(old))
        f, per = self.scorer(), dict(zip(map(str, self.val.days), self.val.period, strict=True))
        f.many([list(new), list(old)])
        return paired_daily(self._cache[frozenset(new)], self._cache[frozenset(old)], per)

    # -- moves ----------------------------------------------------------------------------
    def greedy(self, pool, f, start=(), stop_flat=True):
        """Adds the best member while the set is below k_max (and, `stop_flat`, while the
        best add raises the score at all)."""
        cur = list(start)
        while len(cur) < self.cfg.k_max:
            cand = [j for j in pool if j not in cur]
            if not cand:
                break
            s = f.many([cur + [j] for j in cand])
            if not np.any(np.isfinite(s)):
                break
            best = cand[int(np.nanargmax(s))]
            if stop_flat and cur and not np.nanmax(s) > f(cur):
                break
            cur.append(best)
        return cur

    def swap(self, cur, pool, f, rounds, keep=(), gate=None):
        """Best-improvement one-for-one swaps against `pool`: the best IMPROVING move that clears
        `gate`, not only the best move (LU B2a4: a t 2.93 swap was never taken because a higher
        val move failed the gate first). Members in `keep` (the host) never leave."""
        cur, now = list(cur), f(list(cur))
        for _ in range(rounds):
            moves = [(i, j) for i in range(len(cur)) if cur[i] not in keep for j in pool if j not in cur]
            vals = f.many([cur[:i] + cur[i + 1:] + [j] for i, j in moves])
            took = False
            for v, (i, j) in sorted(zip(vals, moves, strict=True), key=lambda x: -np.nan_to_num(x[0], nan=-9))[:50]:
                if not v > now:
                    break
                new = cur[:i] + cur[i + 1:] + [j]
                if gate is None or gate(new, cur):
                    cur, now, took = new, v, True
                    break
            if not took:
                break
        return cur

    def pool(self, exclude=()):
        c = self.val.moments(self.val.ones())[0]
        return [j for j, n in enumerate(self.names) if np.isfinite(c[j]) and n not in exclude]

    def select(self, starts=(), exclude=(), counts=None):
        """-> member names: the fill + gated swaps from empty (the host alone in host mode) and
        from each start, the best on val; the first given start's set (the incumbent) is replaced
        by another only at the swap bar (LU S7a: a path switch of +0.003 val lost 0.0185 test)."""
        f, t = self.scorer(counts), self.t(counts)
        pool = self.pool(exclude)
        host = (self.val.idx[self.cfg.host],) if self.cfg.host else ()
        sets = [host] + [host + tuple(j for j in self.ix(s) if j not in host)[:self.cfg.k_max - len(host)]
                         for s in starts if s and all(n in self.val.idx for n in s)]
        got = [self.swap(self.greedy(pool, f, start=st), pool, f, self.cfg.k_max, keep=host,
                         gate=lambda a, b: t(a, b) >= self.cfg.swap_t) for st in sets]
        best = max(got, key=lambda s: np.nan_to_num(f(s), nan=-9))
        if len(got) > 1 and best is not got[1] and t(best, got[1]) < self.cfg.swap_t:
            best = got[1]
        return [self.names[j] for j in best]

    # -- the agent's honest estimates (IC only) ------------------------------------------
    def crossval(self, base, recent=()):
        """On each half of the val days select from `base` WITH and WITHOUT the `recent`
        models and score on the other half: latest_models_value = what the last batch adds,
        bank_value = what the bank adds over `base`."""
        if self.cfg.objective != "ic":
            return {"error": f"crossval needs the IC moment form; the selection is {self.cfg.objective}"}
        h = self.val.halves()
        latest, bank = [], []
        for a, b in ((0, 1), (1, 0)):
            fb = self.scorer(h[b])
            with_ = fb(self.ix(self.select([base], counts=h[a])))
            without = fb(self.ix(self.select([base], exclude=set(recent), counts=h[a])))
            latest.append(with_ - without)
            bank.append(with_ - (fb(self.ix(base)) if base else 0.0))
        return {"latest_models_value": float(np.mean(latest)), "latest_folds": [round(x, 5) for x in latest],
                "bank_value": float(np.mean(bank)), "bank_folds": [round(x, 5) for x in bank],
                "n_recent": len(recent)}

    def search(self, F, starts=8, seed=0, top=3):
        """Local search over ENSEMBLES from empty, F and `starts` random sets, swapped to a val
        optimum (no gate); with the held-out gain of searching vs the plain fill (half the days
        to search, the other to score)."""
        if self.cfg.objective != "ic":
            return {"error": f"search needs the IC moment form; the selection is {self.cfg.objective}"}
        pool, rng, t, Fj = self.pool(), np.random.default_rng(seed), self.t(), self.ix(F)
        k = min(self.cfg.k_max, len(pool))

        def run(counts, with_f=True):
            f = self.scorer(counts)
            inits = [[]] + ([Fj] if with_f else []) + [list(rng.choice(pool, k, replace=False))
                                                       for _ in range(starts)]
            return [self.swap(self.greedy(pool, f, start=s) if len(s) < k else list(s), pool, f,
                              4 * self.cfg.k_max) for s in inits], f
        got, f = run(None)
        h, held = self.val.halves(), []
        for a, b in ((0, 1), (1, 0)):
            fa, fb = self.scorer(h[a]), self.scorer(h[b])
            sets, _ = run(h[a], with_f=False)
            held.append(fb(max(sets, key=fa)) - fb(self.greedy(pool, fa)))
        seen, rows = set(), []
        for jj in sorted(got, key=f, reverse=True):
            if frozenset(jj) in seen:
                continue
            seen.add(frozenset(jj))
            rows.append({"members": [self.names[j] for j in jj], "val": round(float(f(jj)), 5),
                         "t_vs_F": round(float(t(jj, Fj)), 2), "shares_with_F": len(set(jj) & set(Fj))})
            if len(rows) == top:
                break
        return {"F_val": round(float(f(Fj)), 5), "sets": rows,
                "held_out_gain_of_searching": round(float(np.mean(held)), 5),
                "held_out_folds": [round(float(x), 5) for x in held]}

    def gains(self, F, target="marginal"):
        """{member: its marginal gain to F} (a member of F: what dropping it loses); `target`
        "best_swap": a non-member's best one-for-one swap into a full F."""
        f, Fj = self.scorer(), self.ix(F)
        base = f(Fj)
        others = [j for j, n in enumerate(self.names) if n != self.cfg.host and j not in Fj]
        drop = f.many([[x for x in Fj if x != j] for j in Fj])
        add = f.many([Fj + [j] for j in others])
        out = {self.names[j]: base - v for j, v in zip(Fj, drop, strict=True) if self.names[j] != self.cfg.host}
        out.update({self.names[j]: v - base for j, v in zip(others, add, strict=True)})
        if target == "best_swap" and len(Fj) >= self.cfg.k_max:
            for j in others:
                s = [x for x in f.many([Fj[:i] + Fj[i + 1:] + [j] for i in range(len(Fj))]) if np.isfinite(x)]
                out[self.names[j]] = max(s) - base if s else float("nan")
        return out
