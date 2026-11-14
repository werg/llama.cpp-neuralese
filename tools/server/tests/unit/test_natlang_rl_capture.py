import base64
import os
import struct

import numpy as np

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
    # a capture needs a positive temperature, so the argmax is the head of each sample's untruncated top-k receipt
    server.start()
    drawn = []
    for seed in (7, 8):
        res = capture(top_p=1e-6, max_tokens=4, seed=seed)
        assert res.status_code == 200, res.body
        samples = res.body["x_natlang_rl_capture"]["samples"]
        assert samples
        for sample in samples:
            ids = struct.unpack(f"<{len(base64.b64decode(sample['top_k']['ids'])) // 4}i", base64.b64decode(sample["top_k"]["ids"]))
            logps = np.frombuffer(base64.b64decode(sample["top_k"]["logps"]), dtype="<f2")
            assert sample["token_id"] == ids[0]
            # the untruncated logp of the argmax token, not the nucleus's 0
            assert sample["logp"] < 0 and abs(sample["logp"] - float(logps[0])) < 1e-2
        drawn.append([s["token_id"] for s in samples])
    assert drawn[0] == drawn[1]
