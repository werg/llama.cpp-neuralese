import threading
import time

import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()

# POST /v1/natlang/score: the log-probability of each reply of an item, the replies of one item scored as a group (a parent
# slot evaluates the shared prefix and the replies continue from it).
CONTINUATIONS = ["Once", "Lily", "There was a little girl"]
ITEM = {"messages": [{"role": "user", "content": "Tell a story."}], "continuations": CONTINUATIONS}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_ctx = 32768
    server.n_batch = 512


def score(item=ITEM, timeout=None):
    res = server.make_request("POST", "/v1/natlang/score", data={"items": [item]}, timeout=timeout)
    assert res.status_code == 200, res.body
    result = res.body["results"][0]
    assert "error" not in result, result
    return result


class SlotHolder:
    """Keeps one slot busy at every moment with single-slot requests, as a second client of a shared server does: the next
    request is streaming before the current one ends, so the slot never goes idle between them."""

    def __init__(self):
        self.stop = threading.Event()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def open(self):
        response = requests.post(server.make_url("/completion"), stream=True,
                                 json={"prompt": "Once upon a time", "n_predict": 8000, "ignore_eos": True, "stream": True})
        lines = response.iter_lines()
        next(lines)  # the first chunk: the request holds its slot
        return lines

    def run(self):
        current = self.open()
        self.started.set()
        while not self.stop.is_set():
            following = self.open()  # a second slot, until `current` ends
            for _ in current:
                pass
            current = following
        for _ in current:
            pass

    def __enter__(self):
        self.thread.start()
        assert self.started.wait(30)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(60)


def test_score_group_of_replies():
    global server
    server.start()
    result = score()
    assert len(result["log_probs"]) == len(CONTINUATIONS)
    assert all(value < 0 for value in result["log_probs"])
    # a reply scores the same alone as inside the group
    for reply, together in zip(CONTINUATIONS, result["log_probs"]):
        alone = score({**ITEM, "continuations": [reply]})
        assert alone["log_probs"][0] == pytest.approx(together, abs=1e-3)


def test_score_group_is_not_starved_by_single_slot_requests():
    global server
    server.start()
    idle = score()
    with SlotHolder():
        # The group needs both slots and only one is ever idle: it must start with the replies that fit and queue the rest,
        # not wait for every slot to be free at once. (Before the fix this request never returned.)
        busy = score(timeout=60)
    assert busy["log_probs"] == pytest.approx(idle["log_probs"], abs=1e-3)
    assert busy["tokens"] == idle["tokens"]


def test_score_does_not_wait_for_a_generation_in_another_slot():
    global server
    server.start()
    idle = score({**ITEM, "continuations": CONTINUATIONS[:1]})
    finished = []

    def generate():
        response = requests.post(server.make_url("/completion"),
                                 json={"prompt": "Once upon a time", "n_predict": 8000, "ignore_eos": True})
        assert response.status_code == 200
        finished.append(time.monotonic())

    generation = threading.Thread(target=generate)
    generation.start()
    time.sleep(0.3)  # the generation holds one of the two slots
    # A score reads the logits of its own prompt: it shares a batch with the generating slot, so it does not wait for the
    # generation to end. (Before the fix a score was kept out of every batch while another slot generated.)
    busy = score({**ITEM, "continuations": CONTINUATIONS[:1]}, timeout=60)
    scored_at = time.monotonic()
    generation.join(60)
    assert finished, "the generation did not end"
    assert scored_at < finished[0], "the score waited for the generation to end"
    assert busy["log_probs"] == pytest.approx(idle["log_probs"], abs=1e-3)
