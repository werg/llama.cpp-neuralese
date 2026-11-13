import os

import pytest
from utils import *

# Captured rollouts need a natlang GGUF (its native identity and producer adapter: natlang's export/native_gguf.py).
NATLANG_GGUF = os.environ.get("LLAMA_NATLANG_GGUF")
pytestmark = pytest.mark.skipif(not NATLANG_GGUF, reason="set LLAMA_NATLANG_GGUF to a natlang GGUF export")
server = ServerPreset.tinyllama2()

# x_natlang_rl_capture under natlang's RL sampling law (temperature 1, top-p 0.95; natlang common/sampling.py): the draw
# comes from the nucleus and each sample records the untruncated post-temperature log-probability, as the native engine
# and the SGLang actor do.


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = NATLANG_GGUF


def capture(**extra):
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "Tell a story."}], "max_tokens": 12, "temperature": 1.0, "seed": 7,
        "x_natlang_rl_capture": True, **extra})
    return res


def test_top_p_capture_records_the_nucleus_law():
    server.start()
    res = capture(top_p=0.95)
    assert res.status_code == 200, res.body
    law = res.body["x_natlang_rl_capture"]["sampling_law"]
    assert law["top_p"] == 0.95 and law["probabilities"] == "post-temperature-untruncated"
    samples = res.body["x_natlang_rl_capture"]["samples"]
    assert samples and all(s["logp"] <= 0 for s in samples)


def test_capture_without_top_p_keeps_the_normalized_law():
    server.start()
    res = capture()
    assert res.status_code == 200, res.body
    law = res.body["x_natlang_rl_capture"]["sampling_law"]
    assert "top_p" not in law and law["probabilities"] == "post-processing-normalized"


def test_a_tiny_nucleus_draws_the_most_likely_token():
    server.start()
    res = capture(top_p=1e-6, max_tokens=4)
    assert res.status_code == 200, res.body
    for sample in res.body["x_natlang_rl_capture"]["samples"]:
        # the untruncated logp of the argmax token: the nucleus held only it
        assert sample["logp"] <= 0
    greedy = capture(temperature=0.0, max_tokens=4)
    assert [s["token_id"] for s in res.body["x_natlang_rl_capture"]["samples"]] == \
           [s["token_id"] for s in greedy.body["x_natlang_rl_capture"]["samples"]]
