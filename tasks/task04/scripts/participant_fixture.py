"""Organizer-only scripted provider; Each valid fixture performs one public-seed single-point smoke job."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import jsonschema


def sample(schema, document):
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise ValueError("external reference must be resolved by caller")
        value = document
        for part in ref[2:].split("/"):
            value = value[part]
        return sample(value, document)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        values = schema["enum"]
        for preferred in [
            "partial",
            "not_determined",
            "not_evaluated",
            "insufficient_evidence",
            "unknown",
        ]:
            if preferred in values:
                return preferred
        return values[0]
    if "oneOf" in schema or "anyOf" in schema:
        choices = schema.get("oneOf", schema.get("anyOf"))
        null = next((x for x in choices if x.get("type") == "null"), None)
        return sample(null or choices[0], document)
    kind = schema.get("type")
    if isinstance(kind, list):
        if "null" in kind:
            return None
        kind = kind[0]
    if kind == "null":
        return None
    if kind == "object":
        return {
            k: sample(schema["properties"][k], document)
            for k in schema.get("required", [])
        }
    if kind == "array":
        return [
            sample(schema["items"], document) for _ in range(schema.get("minItems", 0))
        ]
    if kind in {"number", "integer"}:
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    return "Not determined; this is a protocol wiring fixture."


def outputs(public):
    result = {}
    for name in json.loads((public / ".chembench_agent.json").read_text())[
        "required_artifacts"
    ]:
        if name.endswith(".json"):
            candidates = [
                public / "schema" / name.replace(".json", ".schema.json"),
                public / name.replace(".json", ".schema.json"),
            ]
            path = next(p for p in candidates if p.exists())
            schema = json.loads(path.read_text())
            value = sample(schema, schema)
            if "limitations" in value:
                value["limitations"] = [
                    "Protocol fixture: one single-point job only.",
                    "No conformer search or redox conclusion is claimed.",
                ]
            jsonschema.validate(value, schema)
            result[name] = json.dumps(value)
        else:
            result[name] = (
                "# Protocol wiring fixture\nOne public-seed single-point job only. Scientific conclusions remain undetermined.\n"
            )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--mode", choices=["valid", "invalid"], required=True)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    cfg = json.loads((args.public / ".chembench_agent.json").read_text())
    data = outputs(args.public)
    if args.mode == "invalid":
        data["final_answer.json"] = '{"illegal_candidate":true}'
    calls = []
    for name, content in data.items():
        arguments = {"path": name, "content": content}
        if cfg["task_id"] != "task_05":
            arguments["root"] = "submission"
        calls.append({"tool": "write_file", "arguments": arguments})
    calls.append(
        {
            "tool": "submit",
            "arguments": {"status": "partial"} if cfg["task_id"] == "task_05" else {},
        }
    )
    response = {
        "id": "organizer-fixture",
        "model": "organizer/scripted-provider-v1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"tool_calls": calls}),
                },
            }
        ],
    }

    smoke_script = "from pathlib import Path\nassert not Path('/task').exists()\nassert not Path('/organizer_evidence').exists()\nimport ctypes\nclass Header(ctypes.Structure):\n    _fields_=[('version',ctypes.c_uint32),('pid',ctypes.c_int)]\nclass Data(ctypes.Structure):\n    _fields_=[('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]\nh=Header(0x20080522,0)\nd=(Data*2)()\nassert ctypes.CDLL(None).capget(ctypes.byref(h),d)==0\nassert all(x.effective==0 and x.permitted==0 for x in d)\nprint('SANDBOX_OK')\n"
    script_args = {"path": "wiring_smoke.py", "content": smoke_script}
    python_args = {"script_path": "wiring_smoke.py", "args": []}
    if cfg["task_id"] != "task_05":
        script_args["root"] = "workspace"
        python_args = {"path": "wiring_smoke.py"}
    smoke_calls = [
        {"tool": "write_file", "arguments": script_args},
        {"tool": "run_python", "arguments": python_args},
    ]
    if cfg["task_id"] == "task_05":
        smoke_calls += [
            {
                "tool": "write_file",
                "arguments": {
                    "path": "wiring_water.xyz",
                    "content": (args.public / "data/inputs/monomer_B.xyz").read_text(),
                },
            },
            {
                "tool": "xtb_run",
                "arguments": {
                    "geometry": "wiring_water.xyz",
                    "arguments": ["--gfn", "2", "--chrg", "0", "--uhf", "0"],
                    "namespace": "wiring-smoke",
                    "timeout_seconds": 60,
                },
            },
        ]

    if cfg["task_id"] != "task_05":
        seed = next((args.public / "input").glob("*.xyz"))
        smoke_calls += [
            {
                "tool": "write_file",
                "arguments": {
                    "root": "workspace",
                    "path": "wiring_seed.xyz",
                    "content": seed.read_text(),
                },
            },
            {
                "tool": "xtb_run",
                "arguments": {
                    "operation": "sp",
                    "input_path": "wiring_seed.xyz",
                    "charge": 0,
                    "uhf": 0,
                },
            },
        ]

    class Handler(BaseHTTPRequestHandler):
        requests_seen = 0

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            current = response
            if args.mode == "valid":
                if Handler.requests_seen == 0:
                    current = json.loads(json.dumps(response))
                    current["choices"][0]["message"]["content"] = json.dumps(
                        {"tool_calls": smoke_calls}
                    )
                else:
                    if body.get("tools"):
                        results = [
                            json.loads(m["content"])
                            for m in body["messages"]
                            if m["role"] == "tool"
                        ]
                    else:
                        results = next(
                            json.loads(m["content"])["results"]
                            for m in reversed(body["messages"])
                            if m["role"] == "user"
                            and m["content"].startswith("{")
                            and "results" in json.loads(m["content"])
                        )
                    python_result = next(
                        r for r in results if r.get("tool") == "run_python"
                    )
                    if "SANDBOX_OK" not in python_result.get("result", {}).get(
                        "stdout", ""
                    ):
                        raise RuntimeError(
                            "RUN_PYTHON_SANDBOX_SMOKE_FAILED:"
                            + json.dumps(python_result)
                        )
                    if cfg["task_id"] in {
                        "task_04",
                        "task_05",
                        "task_06",
                        "task_18_redox_reorganization",
                    }:
                        job = next(
                            r["result"] for r in results if r.get("tool") == "xtb_run"
                        )
                        if job.get("status") != "completed" or not job.get("job_id"):
                            raise RuntimeError("XTB_SMOKE_FAILED:" + json.dumps(job))
                        for call in calls:
                            if (
                                call["tool"] == "write_file"
                                and call["arguments"]["path"] == "final_answer.json"
                            ):
                                answer = json.loads(call["arguments"]["content"])
                                if "candidate_results" in answer:
                                    answer["candidate_results"][0][
                                        "evidence_job_ids"
                                    ] = [job["job_id"]]
                                if "evidence_job_ids" in answer:
                                    answer["evidence_job_ids"] = [job["job_id"]]
                                call["arguments"]["content"] = json.dumps(answer)
                        current["choices"][0]["message"]["content"] = json.dumps(
                            {"tool_calls": calls}
                        )
            Handler.requests_seen += 1
            if body.get("tools"):
                current = json.loads(json.dumps(current))
                message = current["choices"][0]["message"]
                action = json.loads(message["content"])
                message.update(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": f"fixture_{Handler.requests_seen}_{i}",
                            "type": "function",
                            "function": {
                                "name": call["tool"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for i, call in enumerate(action["tool_calls"])
                    ],
                )
                current["choices"][0]["finish_reason"] = "tool_calls"
            encoded = json.dumps(current).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
