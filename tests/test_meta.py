import json

from autodata.meta import MetaOptimizer
from autodata.models import SourceDocument, TaskSpec


class Mutator:
    def complete(self, prompt, *, system=""):
        return json.dumps({"revision": "Require paper-specific technical trade-offs.", "rationale": "Avoid generic questions."})


class Evaluator:
    def score(self, spec, sources):
        return (1.0 if "paper-specific" in spec.instructions else 0.5, "questions were too generic")


def test_meta_optimizer_accepts_only_held_out_improvement():
    optimizer = MetaOptimizer(Mutator(), Evaluator())
    spec = TaskSpec("cs", "qa", "Generate research questions")
    improved, population = optimizer.optimize(spec, [SourceDocument("train", "x")], [SourceDocument("val", "x")], iterations=1)
    assert "paper-specific" in improved.instructions
    assert len(population) == 2
    assert optimizer.history[0].accepted
