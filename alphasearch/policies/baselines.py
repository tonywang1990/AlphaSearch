"""The controls: the same harness and budget, no memory, no LLM."""
from alphasearch.records import Plan


class SurrogatePolicy:
    name = "surrogate"
    fills_leftover = False

    def __init__(self, cfg):
        self.cfg = cfg

    def plan(self, obs, models, rng, log):
        """The surrogate's Thompson draws over every untrained recipe."""
        names = list(obs.untrained)
        pick = [names[j] for j in obs.sur.thompson([obs.untrained[n] for n in names], models, rng=rng)]
        return Plan.parse({"experiments": [{"id": "s", "action": "train", "recipes": pick,
                                            "hypothesis": "surrogate Thompson draws"}]})

    def reflect(self, obs, plan, lib, log):
        return []


class RandomPolicy(SurrogatePolicy):
    name = "random"

    def plan(self, obs, models, rng, log):
        """Uniform untrained recipes; half of them and a random feature set where the grammar grows."""
        names = list(obs.untrained)
        if obs.desc.feature_set_rule is None:            # no grammar growth: the whole budget is recipes
            return Plan.parse({"experiments": [{"id": "r1", "action": "train", "hypothesis": "random recipes",
                                                "recipes": list(rng.choice(names, min(models, len(names)),
                                                                           replace=False))}]})
        exps = [{"id": "r1", "action": "train", "hypothesis": "random recipes",
                 "recipes": list(rng.choice(names, min(models // 2, len(names)), replace=False))}]
        if obs.feature_sets:
            base = str(rng.choice(sorted(obs.feature_sets)))
            cols = [c for c in obs.columns if c not in set(obs.feature_sets[base])]
            exps.append({"id": "r2", "action": "feature_set", "hypothesis": "random columns",
                         "name": f"rnd_{rng.integers(1e9)}", "base": base,
                         "features": list(rng.choice(cols, min(24, len(cols)), replace=False)),
                         "twins": list(rng.choice(obs.F, min(4, len(obs.F)), replace=False)), "k": models // 2})
        return Plan.parse({"experiments": exps})
