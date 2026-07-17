from autodata.models import SourceDocument, TaskSpec
from autodata.tasks import ADAPTERS


def test_cs_profile_emits_paper_rubric_contract():
    spec = TaskSpec("research", "qa", "Create a research task", profile="cs_research")
    prompt = ADAPTERS["qa"].generation_prompt(spec, SourceDocument("paper", "paper text"), [])
    assert "10–15" in prompt
    assert "entirely new reasoning angle" in prompt


def test_legal_adapter_requires_paper_six_key_rubric_schema():
    spec = TaskSpec("legal", "legal", "Apply law", profile="legal_reasoning")
    raw = '{"question":"Can I do this?", "target_capabilities":{"primary_focus":["analysis"]}, "rubric":[' + ','.join(['{}'] * 15) + '], "capabilities":[]}'
    candidate = ADAPTERS["legal"].parse(raw, spec, SourceDocument("law", "holding"), "c1")
    assert "exact six-key schema" in ADAPTERS["legal"].validate(candidate)[0]
