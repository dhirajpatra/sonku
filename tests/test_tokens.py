from app.llm import prompts
from app.llm.meta_client import MetaClient


def test_system_prompt_is_stable_and_clock_only_when_asked(rt):
    a = prompts.system_prompt("en", "Asha", "likes tea")
    assert "Current time" not in a and a == prompts.system_prompt("en", "Asha", "likes tea")
    assert prompts.time_note("what time is it") and not prompts.time_note("tell me a joke")
    prompt = MetaClient.flat_prompt(a, [("user", "hi"), ("assistant", "hello")], "what time is it")
    assert prompt.index("hello") < prompt.index("Current time") < prompt.rindex("User: what time is it")


def test_usage_counted_even_for_empty_replies(rt):
    c = rt.router.meta
    c._account({"usage": {"input_tokens": 120, "output_tokens": 200,
                          "input_tokens_details": {"cached_tokens": 100}, "output_tokens_details": {"reasoning_tokens": 200}}})
    c._account({})  # provider sent no usage block: must not crash
    assert c.totals["input"] == 120 and c.totals["cached"] == 100 and c.totals["reasoning"] == 200 and c.totals["calls"] == 2
    assert rt.router.status()["meta_usage"]["output"] == 200


def test_history_trimmed_in_blocks(rt):
    from tests.test_orchestrator import stub_llm
    rt.s.history_turns = 2  # limit 4 messages
    stub_llm(rt)
    seen = []
    for i in range(6):
        rt.orch.handle_utterance(("sonku " if i == 0 else "") + f"question number {i}", "en")
        seen.append(len(rt.orch.session.history))
    assert max(seen) <= 4 and seen[:2] == [2, 4]
