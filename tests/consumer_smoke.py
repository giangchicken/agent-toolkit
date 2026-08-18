#!/usr/bin/env python3
"""The consumer contract of agent-toolkit 0.1.0, checked against an install.

Run this against an *installed wheel*, not the source tree::

    python3.12 -m venv /tmp/at-full
    /tmp/at-full/bin/pip install "dist/agent_toolkit-0.1.0-py3-none-any.whl[llm]"
    /tmp/at-full/bin/python tests/consumer_smoke.py

It imports the fifteen symbols that ``docs/agent-toolkit/plan.md`` T11 names as
the pipeline's dependency and calls each one once. ``EXERCISED`` is compared
against ``CONTRACT`` at the end, so a symbol that imports but is never called
fails the run: the claim is that the contract works, not that it resolves.

The two LLM entry points are pointed at a stub ``http.server`` on localhost
rather than mocked. Every test under ``tests/`` replaces
``httpx2.AsyncHTTPTransport.handle_async_request``, so the suite has never
exercised the transport that the installed ``openai`` actually uses. This script
is the only place that does, and it is the right place: against the wheel, with
the resolved dependency versions, on the consumer's interpreter.

Two things it needs from the environment. A free localhost port, for the stub.
And, the first time it runs on a machine, network access to fetch tiktoken's
``cl100k_base`` vocabulary -- ``count_tokens`` cannot answer without it. Set
``TIKTOKEN_CACHE_DIR`` to a pre-populated directory on a host that has no egress.

Not a pytest module. It has to run in an environment that has the wheel and
nothing else.
"""

import asyncio
import hashlib
import json
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CONTRACT = frozenset(
    {
        "LLMError hierarchy",
        "TrafficController",
        "complete",
        "complete_structured",
        "compute_hash",
        "count_tokens",
        "extract_json_from_text",
        "iter_json_array_file",
        "model_family",
        "normalize_text",
        "read_json",
        "read_jsonlines",
        "slot_filling",
        "write_json",
        "write_jsonlines",
    }
)

EXERCISED: set[str] = set()


def check(name: str, actual: object, expected: object) -> None:
    """Record ``name`` as exercised, or exit non-zero saying how it differed."""
    if actual != expected:
        raise SystemExit(
            f"FAIL  {name}\n  expected {expected!r}\n  got      {actual!r}"
        )
    EXERCISED.add(name)
    print(f"  ok  {name}")


# --------------------------------------------------------------------------
# The stub endpoint
# --------------------------------------------------------------------------


class _Stub(BaseHTTPRequestHandler):
    """Answers one chat completion with whatever ``reply`` currently holds."""

    reply = ""
    seen: list[dict[str, object]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        _Stub.seen.append(request)
        body = json.dumps(
            {
                "id": "smoke",
                "object": "chat.completion",
                "created": 0,
                "model": request.get("model", "stub"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": _Stub.reply},
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def closed_port() -> int:
    """A port nothing is listening on, for the connection-failure path."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# --------------------------------------------------------------------------
# The three groups
# --------------------------------------------------------------------------


def check_string_utils() -> None:
    from agent_toolkit.string_utils import (
        compute_hash,
        extract_json_from_text,
        normalize_text,
        slot_filling,
    )

    check(
        "slot_filling",
        slot_filling("Xin chào {{name}}", {"name": "Dũng"}),
        "Xin chào Dũng",
    )
    # The jury's shape: a fenced array in prose, which plain json.loads rejects.
    check(
        "extract_json_from_text",
        extract_json_from_text('Vote:\n```json\n["get_weather"]\n```\n'),
        ["get_weather"],
    )
    check("normalize_text", normalize_text("  Xin  chào  "), "Xin chào")
    check(
        "compute_hash",
        compute_hash("dataforce"),
        hashlib.sha256(b"dataforce").hexdigest(),
    )


def check_file_utils() -> None:
    from agent_toolkit.file_utils import (
        iter_json_array_file,
        read_json,
        read_jsonlines,
        write_json,
        write_jsonlines,
    )

    records = [{"id": 1, "text": "Xin chào"}, {"id": 2, "text": "Tạm biệt"}]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        write_json(root / "records.json", records)
        check("write_json", (root / "records.json").exists(), True)
        check("read_json", read_json(root / "records.json"), records)

        write_jsonlines(root / "records.jsonl", records)
        check("write_jsonlines", (root / "records.jsonl").exists(), True)
        check("read_jsonlines", read_jsonlines(root / "records.jsonl"), records)

        # The one the 126 MiB corpus depends on: the file is never read whole.
        check(
            "iter_json_array_file",
            [row["id"] for row in iter_json_array_file(root / "records.json")],
            [1, 2],
        )


def check_llm(base_url: str) -> None:
    from agent_toolkit.llm import (
        TrafficController,
        complete,
        complete_structured,
        count_tokens,
        model_family,
    )
    from agent_toolkit.llm.exceptions import LLMAPIError, LLMError
    from agent_toolkit.llm.retry import RetryPolicy

    # What the jury uses it for: distinct families, counted.
    check(
        "model_family",
        sorted({model_family(n) for n in ("gemma-4-31B-it", "Qwen3-8B", "glm-5.1")}),
        ["gemma", "glm", "qwen"],
    )
    check("count_tokens", count_tokens([{"role": "user", "content": "hi"}]) > 0, True)

    async def controller_admits_one() -> int:
        controller = TrafficController("smoke", max_concurrency=1)
        async with controller:
            return controller.active_requests

    check("TrafficController", asyncio.run(controller_admits_one()), 1)

    # A reasoning model's shape, so this proves the answer arrives unpolluted
    # rather than only that a string came back.
    _Stub.reply = "<think>Chào hỏi thôi.</think>Xin chào!"
    check(
        "complete",
        asyncio.run(
            complete("Chào", model="stub-model", base_url=base_url, api_key="none")
        ),
        "Xin chào!",
    )

    _Stub.reply = 'Vote:\n```json\n["get_weather"]\n```'
    schema = {
        "type": "array",
        "items": {"type": "string", "enum": ["get_weather", "book_flight"]},
    }
    value, info = asyncio.run(
        complete_structured(
            "Which tool?",
            schema,
            model="stub-model",
            base_url=base_url,
            api_key="none",
        )
    )
    check(
        "complete_structured",
        (value, info.ok, info.repaired),
        (["get_weather"], True, True),
    )

    # The hierarchy as a consumer meets it: one `except LLMError` around the call.
    try:
        asyncio.run(
            complete(
                "Chào",
                model="stub-model",
                base_url=f"http://127.0.0.1:{closed_port()}/v1",
                api_key="none",
                retry=RetryPolicy(max_retries=0),
            )
        )
    except LLMError as exc:
        check(
            "LLMError hierarchy",
            (type(exc).__name__, isinstance(exc, LLMAPIError), exc.status_code),
            ("LLMAPIError", True, None),
        )
    else:
        raise SystemExit("FAIL  LLMError hierarchy: a dead port did not raise")


def main() -> None:
    import agent_toolkit

    if agent_toolkit.__version__ != "0.1.0":
        raise SystemExit(f"FAIL  __version__ is {agent_toolkit.__version__!r}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"

    try:
        check_string_utils()
        check_file_utils()
        check_llm(base_url)
    finally:
        server.shutdown()

    missing = CONTRACT - EXERCISED
    if missing:
        raise SystemExit(f"FAIL  never exercised: {sorted(missing)}")
    extra = EXERCISED - CONTRACT
    if extra:
        raise SystemExit(f"FAIL  not in the contract: {sorted(extra)}")

    print(f"\n{len(CONTRACT)}/{len(CONTRACT)} consumer symbols exercised")
    print(f"{len(_Stub.seen)} requests reached the stub endpoint")


if __name__ == "__main__":
    main()
