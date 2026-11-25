import json
import os
import struct

import pytest
import requests
from utils import *

# Native states on a natlang GGUF (its native identity, role markers and producer adapter: natlang's export/native_gguf.py).
NATLANG_GGUF = os.environ.get("LLAMA_NATLANG_GGUF")
pytestmark = pytest.mark.skipif(not NATLANG_GGUF, reason="set LLAMA_NATLANG_GGUF to a natlang GGUF export")
server = ServerPreset.tinyllama2()

# A produce span is stored with producer_kind "span" (natlang model/state_block.py), and a state of this server's identity
# with producer_kind "text" or "span" (no adapter binding, no derivation) is read as its own tokens inline where the part
# stands, with no consume markers: receipt mode in-context-text, recorded by the RL capture as tokens (natlang
# serve/engine.py reads_in_context). NATLANG_NEURALESE_SPAN_IN_CONTEXT=0 splices it as stored. A request's
# x_natlang_producer_kind "capture" stores its span as a capture (a reply captured in its own context), which splices
# as stored unless NATLANG_NEURALESE_CAPTURE_IN_CONTEXT=1.


@pytest.fixture(autouse=True)
def create_server(monkeypatch):
    global server
    monkeypatch.delenv("NATLANG_NEURALESE_SPAN_IN_CONTEXT", raising=False)
    monkeypatch.delenv("NATLANG_NEURALESE_TEXT_IN_CONTEXT", raising=False)
    monkeypatch.delenv("NATLANG_NEURALESE_CAPTURE_IN_CONTEXT", raising=False)
    server = ServerPreset.tinyllama2()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = NATLANG_GGUF
    server.jinja = True
    server.n_ctx = 2048
    server.n_slots = 1


def produce(**extra):
    info = server.make_request("GET", "/v1/neuralese/info")
    assert info.status_code == 200, info.body
    assert "in-context-text" in info.body["native_state"]["consumption_modes"]
    assert "native.producer-kind.capture" in info.body["capabilities"]
    po, pc = info.body["native_state"]["role_marker_ids"][:2]
    grammar = f'root ::= "Billing: " <[{po}]> " refund request for order 4411" <[{pc}]> " done"'
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "Summarize the ticket."}], "max_tokens": 40, "temperature": 0,
        "grammar": grammar, **extra})
    assert res.status_code == 200, res.body
    return res.body["native_state"]["blocks"][0]["id"], info.body["native_state"]["role_marker_ids"]


def stored_token_ids(sid):
    body = requests.get(server.make_url(f"/v1/neuralese/native-states/{sid}")).content
    (size,) = struct.unpack("<Q", body[:8])
    header = json.loads(body[8:8 + size])
    start, end = header["token_ids"]["data_offsets"]
    data = body[8 + size + start:8 + size + end]
    return list(struct.unpack(f"<{len(data) // 8}q", data))


def consume(sid, **extra):
    return server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Which order is this about? "},
                                                  {"type": "native_state", "id": sid}]}],
        "max_tokens": 4, "temperature": 1.0, "seed": 7, **extra})


def test_a_produce_span_carries_producer_kind_span():
    server.start()
    sid, _ = produce()
    meta = server.make_request("GET", f"/v1/neuralese/native-states/{sid}/meta")
    assert meta.status_code == 200, meta.body
    assert meta.body["producer_kind"] == "span"
    assert "derivation" not in meta.body and "adapter_binding" not in meta.body


def test_a_produce_span_is_read_in_context():
    server.start()
    sid, markers = produce()
    ids = stored_token_ids(sid)
    assert ids
    res = consume(sid, x_natlang_rl_capture=True, top_p=0.95)
    assert res.status_code == 200, res.body
    imports = res.body["native_state"]["imports"]
    assert imports == [{"id": sid, "mode": "in-context-text", "observed_native_state_id": sid}]
    prompt = res.body["x_natlang_rl_capture"]["prompt"]
    assert all(segment["kind"] == "tokens" for segment in prompt)
    tokens = [t for segment in prompt for t in segment["token_ids"]]
    assert not set(markers[2:]) & set(tokens), "no consume markers around an in-context read"
    assert any(tokens[i:i + len(ids)] == ids for i in range(len(tokens))), "the state's own tokens inline"


def test_the_span_opt_out_splices_as_stored(monkeypatch):
    monkeypatch.setenv("NATLANG_NEURALESE_SPAN_IN_CONTEXT", "0")
    server.start()
    sid, _ = produce()
    res = consume(sid, x_natlang_rl_capture=True, top_p=0.95)
    assert res.status_code == 200, res.body
    assert [item["mode"] for item in res.body["native_state"]["imports"]] == ["exact-native-state"]
    prompt = res.body["x_natlang_rl_capture"]["prompt"]
    assert [segment["id"] for segment in prompt if segment["kind"] == "native_state"] == [sid]


def test_the_text_opt_out_leaves_spans_in_context(monkeypatch):
    monkeypatch.setenv("NATLANG_NEURALESE_TEXT_IN_CONTEXT", "0")
    server.start()
    sid, _ = produce()
    res = consume(sid)
    assert res.status_code == 200, res.body
    assert [item["mode"] for item in res.body["native_state"]["imports"]] == ["in-context-text"]


def test_a_capture_request_stores_producer_kind_capture_and_splices_it():
    server.start()
    sid, _ = produce(x_natlang_producer_kind="capture")
    meta = server.make_request("GET", f"/v1/neuralese/native-states/{sid}/meta")
    assert meta.status_code == 200, meta.body
    assert meta.body["producer_kind"] == "capture"
    span_id, _ = produce()
    assert span_id != sid, "the kind is part of the content id"
    res = consume(sid, x_natlang_rl_capture=True, top_p=0.95)
    assert res.status_code == 200, res.body
    assert [item["mode"] for item in res.body["native_state"]["imports"]] == ["exact-native-state"]
    prompt = res.body["x_natlang_rl_capture"]["prompt"]
    assert [segment["id"] for segment in prompt if segment["kind"] == "native_state"] == [sid]


def test_the_capture_opt_in_reads_it_in_context(monkeypatch):
    monkeypatch.setenv("NATLANG_NEURALESE_CAPTURE_IN_CONTEXT", "1")
    server.start()
    sid, _ = produce(x_natlang_producer_kind="capture")
    res = consume(sid)
    assert res.status_code == 200, res.body
    assert [item["mode"] for item in res.body["native_state"]["imports"]] == ["in-context-text"]


def test_an_unknown_producer_kind_is_refused():
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "x_natlang_producer_kind": "text"})
    assert res.status_code == 400, res.body
