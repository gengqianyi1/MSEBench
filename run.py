#!/usr/bin/env python3
"""Run evaluations from an assembled MSEBench local workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from msebench.config import load_profile, env_dict  # noqa: E402
from msebench.utils import ensure_dir, timestamp  # noqa: E402

HARNESS = ROOT / "tasks"
FRAMEWORKS = ROOT / "frameworks"
HARBOR_RUNNER_IMAGE = "matbench/harbor-runner:20260906"
HARBOR_RUNNER_IMAGE_1417 = "chembench/harbor-runner:2.1.0-r10"
def _env_with_legacy(name, default=None):
    return os.getenv(name) or os.getenv(name.replace('MSEBENCH_', 'MATBENCH_')) or default


RUNS_ROOT = Path(_env_with_legacy("MSEBENCH_RUNS_ROOT", str(ROOT / "runs")))
RESULTS_ROOT = Path(_env_with_legacy("MSEBENCH_RESULTS_ROOT", str(ROOT / "results")))

TASKS = {
    "4": {"family": "msebench_f2", "pkg": "task04"},
    "5": {"family": "msebench_f2", "pkg": "task05"},
    "6": {"family": "msebench_f2", "pkg": "task06"},
    "18": {"family": "msebench_f2", "pkg": "task18"},
    "1": {"family": "msebench_f1", "pkg": "01-03", "harbor_task": "task01"},
    "2": {"family": "msebench_f1", "pkg": "01-03", "harbor_task": "task02"},
    "3": {"family": "msebench_f1", "pkg": "01-03", "harbor_task": "task03"},
    "7": {"family": "msebench_f3", "pkg": "07-09", "organizer_task": 7},
    "8": {"family": "msebench_f3", "pkg": "07-09", "organizer_task": 8},
    "9": {"family": "msebench_f3", "pkg": "07-09", "organizer_task": 9},
    "10": {"family": "msebench_f4", "pkg": "10-13", "runner": "revision6"},
    "11": {"family": "msebench_f4", "pkg": "10-13", "runner": "revision6"},
    "12": {"family": "msebench_f4", "pkg": "10-13", "runner": "revision6"},
    "13": {"family": "msebench_f4", "pkg": "10-13", "runner": "revision6"},
    "14": {"family": "msebench_f5", "pkg": "14-17", "runner": "1417", "task_slug": "task-14-multiobjective-mol-design"},
    "15": {"family": "msebench_f5", "pkg": "14-17", "runner": "1417", "task_slug": "task-15-lipophilic-sulfonamide-edit-budget"},
    "16": {"family": "msebench_f5", "pkg": "14-17", "runner": "1417", "task_slug": "task-16-tablet-microenvironment-cqa"},
    "17": {"family": "msebench_f5", "pkg": "14-17", "runner": "1417", "task_slug": "task-17-spherical-agglomeration"},
    "19": {"family": "msebench_f6", "pkg": "19-20", "runner": "1920"},
    "20": {"family": "msebench_f6", "pkg": "19-20", "runner": "1920"},
    "textqa": {"family": "msebench_f0"},
}

ORGANIZER_IMAGE = "chembench/organizer:harbor-0.21.0-20260913"


def _run(cmd, env, cwd, shell=False):
    merged = dict(os.environ)
    merged.update({k: v for k, v in env.items() if v not in (None, "")})
    return subprocess.run(cmd, cwd=str(cwd), env=merged, shell=shell,
                          capture_output=True, text=True)


def run_chembench(task, cfg, verify=False):
    pkg = HARNESS / TASKS[task]["pkg"]
    env = env_dict(cfg)
    env["CHEMBENCH_LLM_CALL_LOG"] = str(RUNS_ROOT / f"task{task:0>2}" / "llm_calls.jsonl")
    if verify:
        proc = _run(["bash", "run.sh", "smoke"], env, pkg)
        return {"task": task, "family": "msebench_f2", "verify": proc.returncode == 0,
                "stdout_tail": (proc.stdout or "")[-400:]}
    # 2026-09-27 fix: re-probe the gateway before each container launch; overnight outages hit mid-batch long after the one-shot precheck
    if not _gateway_ready(cfg):
        return {"task": task, "family": "msebench_f2", "exit": None,
                "error": "LLM gateway unreachable after mid-flight probes; task skipped"}
    proc = _run(["bash", "run.sh", "model"], env, pkg)
    grader = None
    model_root = RUNS_ROOT / f"task{int(task):02d}" / "model"
    attempts = sorted(model_root.glob("*/deployment_record.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True) if model_root.is_dir() else []
    if attempts:
        try:
            data = json.loads(attempts[0].read_text(encoding="utf-8"))
            grader = {"score": data.get("reward"),
                      "regression_passed": data.get("regression_passed"),
                      "path": str(attempts[0])}
        except Exception as exc:
            grader = {"error": str(exc)}
    return {"task": task, "family": "msebench_f2", "exit": proc.returncode, "grader": grader,
            "stderr_tail": (proc.stderr or "")[-400:]}


def _materialize_harbor_profile(src: Path, cfg: dict, out: Path) -> Path:
    import yaml
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    model = cfg.get("model") or data.get("model_name") or ""
    data["name"] = f"{cfg.get('profile', 'default')}_{''.join(c for c in model if c.isalnum())}"
    data["model_name"] = model
    if cfg.get("api_base"):
        data["api_base_url"] = cfg["api_base"]
    for key in ("temperature", "max_tokens", "top_p", "reasoning_effort", "enable_thinking", "stream",
                "use_native_tool_calls", "api_format"):
        if key in cfg and cfg[key] is not None:
            data[key] = cfg[key]
    if cfg.get("request_timeout") is not None:
        data["request_timeout_sec"] = cfg["request_timeout"]
    if cfg.get("request_retries") is not None:
        data["request_retries"] = cfg["request_retries"]
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def run_harbor(task, cfg, verify=False):
    meta = TASKS[task]
    pkg = FRAMEWORKS / meta["pkg"]
    run_root = ensure_dir(RUNS_ROOT / f"task{task}-{timestamp()}")
    profile_file = "default.profile.yaml"
    profile = _materialize_harbor_profile(
        pkg / "tasks" / "task03" / "profiles" / profile_file,
        cfg,
        run_root / "harbor_profile.yaml",
    )
    cmd = ["docker", "run", "--rm", "--network", "host",
           "-v", "/var/run/docker.sock:/var/run/docker.sock",
           "-v", f"{ROOT}:{ROOT}", "-v", f"{run_root}:{run_root}",
           "-w", str(pkg), "-e", "MSEBENCH_API_KEY",
           "-e", "CHEMBENCH_TEMPERATURE", "-e", "CHEMBENCH_TOP_P",
           "-e", "CHEMBENCH_ENABLE_THINKING", "-e", "CHEMBENCH_REASONING_EFFORT",
           "-e", "CHEMBENCH_MAX_TOKENS",
           "-e", "CHEMBENCH_MAX_AGENT_TURNS", "-e", "CHEMBENCH_TOKEN_BUDGET",
           HARBOR_RUNNER_IMAGE,
           "python", f"{pkg}/common/scripts/run_llm_evaluation.py",
           "--profile", str(profile), "--output-dir", f"{run_root}/model_run",
           "--task", meta["harbor_task"], "--engine", "harbor"]
    if verify:
        cmd.append("--install-only")
    proc = _run(cmd, env_dict(cfg), pkg)
    return {"task": task, "family": "msebench_f1", "exit": proc.returncode,
            "output_dir": str(run_root), "stdout_tail": (proc.stdout or "")[-400:]}


def _organizer_runtime():
    env_root = os.getenv("CHEMBENCH_RUNTIME_ROOT")
    if env_root:
        return Path(env_root)
    pkg_runtime = FRAMEWORKS / TASKS["7"]["pkg"] / "runtime"
    if pkg_runtime.is_symlink():
        return Path(os.path.realpath(pkg_runtime))
    return pkg_runtime


def run_organizer(task, cfg, verify=False):
    meta = TASKS[task]
    pkg = FRAMEWORKS / meta["pkg"]
    env = env_dict(cfg)
    env["CHEMBENCH_ALLOW_PATCHED_SIDECAR"] = "1"
    flags = ["--preflight-only"] if verify else []
    venv = _env_with_legacy("MSEBENCH_HARBOR_VENV")
    if venv:
        env["PATH"] = f"{venv}/bin:" + env.get("PATH", os.environ.get("PATH", ""))
        env.setdefault("CHEMBENCH_HARBOR_BIN", f"{venv}/bin/harbor")
    proc = _run(["bash", "./start_task.sh", "--task", str(meta["organizer_task"]),
                 "--skip-image-load"] + flags, env, pkg)
    audit = None
    # 2026-09-27 fix: verify mode = preflight only; don't surface the last real run's audit as its verdict.
    if not verify:
        runtime = _organizer_runtime()
        matches = sorted(runtime.glob(f"task0{meta['organizer_task']}/*/output/run_audit.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            try:
                data = json.loads(matches[0].read_text(encoding="utf-8"))
                audit = {"status": data.get("status"), "score": data.get("score"),
                         "reward": data.get("reward")}
            except Exception as exc:
                audit = {"error": str(exc)}
    return {"task": task, "family": "msebench_f3", "exit": proc.returncode,
            "audit": audit, "stdout_tail": (proc.stdout or "")[-400:]}


def run_textqa(cfg, concurrency, input_path):
    cmd = [sys.executable, str(ROOT / "tasks" / "textqa" / "run_textqa.py"),
           "--profile", cfg["profile"],
           "--concurrency", str(concurrency)]
    if input_path:
        cmd.extend(["--input", input_path])
    proc = _run(cmd, env_dict(cfg), ROOT)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def run_revision6(task, cfg, verify=False):
    env = env_dict(cfg)
    runs_root = ensure_dir(RUNS_ROOT)
    env["CHEMBENCH_LLM_CALL_LOG"] = str(runs_root / f"task{task:0>2}" / "llm_calls.jsonl")
    mode = "install-only" if verify else "model"
    runner = FRAMEWORKS / "10-13" / "run_task.py"
    cmd = ["docker", "run", "--rm", "--network", "host",
           "-v", "/var/run/docker.sock:/var/run/docker.sock",
           "--mount", f"type=bind,source={ROOT},target={ROOT}",
           "--mount", f"type=bind,source={runs_root},target={runs_root}",
           "--workdir", str(ROOT),
           "--env", f"MSEBENCH_INTEG_ROOT={ROOT}",
           *[x for name in ("MSEBENCH_API_KEY", "CHEMBENCH_API_KEY", "CHEMBENCH_API_BASE", "CHEMBENCH_API_URL",
                            "CHEMBENCH_MODEL_ID", "CHEMBENCH_TEMPERATURE", "CHEMBENCH_TOP_P",
                            "CHEMBENCH_MAX_TOKENS", "CHEMBENCH_STREAM", "CHEMBENCH_ENABLE_THINKING",
                            "CHEMBENCH_REASONING_EFFORT", "CHEMBENCH_REQUEST_TIMEOUT",
                            "CHEMBENCH_REQUEST_RETRIES", "CHEMBENCH_LLM_CALL_LOG",
                             "CHEMBENCH_MAX_AGENT_TURNS")
              if env.get(name) for x in ("--env", name)],
           ORGANIZER_IMAGE, "python", str(runner),
           "--task", task, "--mode", mode, "--output-root", str(runs_root)]
    proc = _run(cmd, env, ROOT)
    grader = None
    model_root = runs_root / f"task{task}" / "model"
    attempts = sorted(model_root.glob("*/deployment_record.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True) if model_root.is_dir() else []
    if attempts:
        try:
            data = json.loads(attempts[0].read_text(encoding="utf-8"))
            grader = {"score": data.get("reward"),
                      "regression_passed": data.get("regression_passed"),
                      "path": str(attempts[0])}
        except Exception as exc:
            grader = {"error": str(exc)}
    return {"task": task, "family": "msebench_f4", "runner": "revision6",
            "exit": proc.returncode, "grader": grader,
            "stdout_tail": (proc.stdout or "")[-400:]}


def run_1417(task, cfg, verify=False):
    meta = TASKS[task]
    pkg = FRAMEWORKS / meta["pkg"]
    env = env_dict(cfg)
    env["CHEMBENCH_LLM_CALL_LOG"] = str(RUNS_ROOT / f"task{task:0>2}" / "llm_calls.jsonl")
    sub = "llm-mock" if verify else "llm"
    cmd = ["docker", "run", "--rm", "--pull", "never",
           "-v", "/var/run/docker.sock:/var/run/docker.sock",
           "-v", f"{ROOT}:{ROOT}",
           "-w", str(pkg),
           "-e", "CHEMBENCH_MODEL_ID", "-e", "CHEMBENCH_API_KEY",
           "-e", "CHEMBENCH_API_BASE", "-e", "CHEMBENCH_API_URL",
           "-e", "CHEMBENCH_TEMPERATURE", "-e", "CHEMBENCH_TOP_P",
           "-e", "CHEMBENCH_MAX_TOKENS", "-e", "CHEMBENCH_ENABLE_THINKING",
           "-e", "CHEMBENCH_REASONING_EFFORT", "-e", "CHEMBENCH_REQUEST_TIMEOUT",
           "-e", "CHEMBENCH_REQUEST_RETRIES", "-e", "CHEMBENCH_STREAM",
           "-e", "CHEMBENCH_LLM_CALL_LOG",
           "-e", "CHEMBENCH_MAX_AGENT_TURNS", "-e", "CHEMBENCH_TOKEN_BUDGET",
HARBOR_RUNNER_IMAGE_1417,
            "./chembench", sub, meta["task_slug"]]
    # 2026-09-27 fix: re-probe the gateway before each container launch; overnight outages hit mid-batch long after the one-shot precheck
    if not verify and not _gateway_ready(cfg):
        return {"task": task, "family": "msebench_f5", "runner": "1417", "exit": None,
                "error": "LLM gateway unreachable after mid-flight probes; task skipped"}
    proc = _run(cmd, env, pkg)
    return {"task": task, "family": "msebench_f5", "runner": "1417",
            "exit": proc.returncode, "stdout_tail": (proc.stdout or "")[-400:]}


def _plan_1920_stale(pkg, campaign):
    plan_file = campaign / "frozen_plan.json"
    if not plan_file.exists():
        return True
    try:
        files = (json.loads(plan_file.read_text()) or {}).get("files") or {}
        for rel, expected in files.items():
            path = pkg / rel
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return True
    except Exception:
        return True
    return False


def _prepare_1920(pkg, campaign, cfg):
    if campaign.exists():
        campaign.rename(campaign.parent / f"{campaign.name}.prev-{timestamp()}")
    script = pkg / "scripts" / "run_llm.py"
    return _run([sys.executable, str(script), "--campaign", str(campaign), "--prepare-only"],
                env_dict(cfg), pkg)


def _campaign_1920(task):
    return Path(os.getenv("MSEBENCH_TASK1920_CAMPAIGN", str(RUNS_ROOT / "task1920" / f"task{task}")))


def run_1920(task, cfg, verify=False):
    pkg = FRAMEWORKS / TASKS[task]["pkg"]
    campaign = _campaign_1920(task)
    script = pkg / "scripts" / "run_llm.py"
    if verify:
        campaign = campaign / ("verify-" + timestamp() + "-" + str(time.time_ns()))
        cmd = [sys.executable, str(script), "--campaign", str(campaign), "--prepare-only"]
    else:
        model_id = (cfg.get("model") or "").lower()
        # Registered models use their frozen key; any other LLM falls back to a
        # generic key derived from the model id (route comes from CHEMBENCH_* env).
        model_key = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in model_id)
        if not model_key:
            raise SystemExit(
                "task 19/20 requires a configured model (cfg['model'] is empty)")
        if _plan_1920_stale(pkg, campaign):
            prep = _prepare_1920(pkg, campaign, cfg)
            if prep.returncode != 0:
                return {"task": task, "family": "msebench_f6", "runner": "1920",
                        "exit": prep.returncode,
                        "stdout_tail": ((prep.stderr or "") + (prep.stdout or ""))[-400:]}
        cell_dir = campaign / f"{model_key}-task{task}"
        if cell_dir.exists():
            cell_dir.rename(campaign / f"{model_key}-task{task}.prev-{timestamp()}")
        cmd = [sys.executable, str(script), "--campaign", str(campaign),
               "--model", model_key, "--task", task]
    proc = _run(cmd, env_dict(cfg), pkg)
    return {"task": task, "family": "msebench_f6", "runner": "1920",
            "exit": proc.returncode, "stdout_tail": (proc.stdout or "")[-400:],
            "stderr_tail": (proc.stderr or "")[-1000:]}


def _gateway_ready(cfg):
    """Cheap LLM ping before a batch burns hours of container time.

    2026-09-22: an overnight antchat outage (02:00-08:00) zeroed six tasks of a
    --task all batch while task01, which ran before the window, was fine. A
    16-token no-think probe costs nothing and turns that into a clean abort.
    """
    from msebench.config import api_key
    from msebench.llm import LLMClient
    probe = dict(cfg)
    probe["api_key"] = api_key(cfg)
    probe["max_tokens"] = 16
    # 不要硬塞 enable_thinking：部分模型（如 gpt-5.6-sol）会 400 unknown_parameter，
    # 导致预检失败 → 每个 task 被 skip。置 None 时 LLMClient 会省略该字段。
    probe["enable_thinking"] = None
    probe["reasoning_effort"] = "low"
    # 2026-09-27 fix: keep tiny; urlopen applies this timeout per resolved IP (host has several).
    probe["request_timeout"] = 5
    for attempt in range(4):
        try:
            LLMClient(probe).chat([{"role": "user", "content": "ping"}])
            return True
        except Exception as exc:
            print(f"[preflight] LLM gateway unreachable (attempt {attempt + 1}/4): {exc}",
                  file=sys.stderr)
            if attempt + 1 < 4:
                time.sleep(60)
    return False


def _task_artifact_dirs(task, family):
    if family == "msebench_f1":
        root, pattern = RUNS_ROOT, f"task{task}-*"
    elif family in {"msebench_f2", "msebench_f4"}:
        root, pattern = RUNS_ROOT, f"task{task.zfill(2)}/model/*"
    elif family == "msebench_f3":
        root, pattern = _organizer_runtime(), f"task{task.zfill(2)}/*"
    elif family == "msebench_f5":
        root, pattern = FRAMEWORKS / "14-17/jobs", f"llm-task-{task}-*"
    elif family == "msebench_f6":
        root, pattern = _campaign_1920(task), f"*-task{task}"
    else:
        return []
    return [path for path in root.glob(pattern) if path.is_dir()]


def _stamp_model(task, cfg, previous):
    """在本题的产物目录与记录文件里标注模型身份，便于区分来源。"""
    meta = TASKS.get(str(task)) or {}
    family = meta.get("family")
    model = cfg.get("model") or ""
    profile = cfg.get("_profile_name") or ""
    dirs = set(_task_artifact_dirs(task, family)) - previous
    if not dirs:
        return
    newest = max(dirs, key=lambda path: path.stat().st_mtime_ns)
    ident = {"task": int(task), "family": family, "model": model, "profile": profile,
             "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        (newest / "run_identity.json").write_text(
            json.dumps(ident, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        return
    for name in ("deployment_record.json", "grader-result.json", "trial_result.json", "result.json", "config.json"):
        for rec in [newest / name] + list(newest.glob(f"*/{name}"))[:2]:
            try:
                data = json.loads(rec.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and not data.get("model"):
                data["model"] = model
                data["model_profile"] = profile
                try:
                    rec.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                except OSError:
                    pass
    print(f"[stamp] task{int(task):02d} model={model} → {newest}")


def _archive_runs():
    """把本批所有家族的运行记录追加归集到指定输出位置（append-only，不覆盖历史）。"""
    tool = Path(__file__).resolve().parent / "tools" / "collect_runs.py"
    if not tool.exists():
        return
    try:
        proc = subprocess.run([sys.executable, str(tool)], capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
            print(f"[archive] FAILED rc={proc.returncode}: {tail[0] if tail else ''}")
            return
        last = (proc.stdout or "").strip().splitlines()[-1:]
        print(f"[archive] {last[0] if last else 'ok'}")
    except Exception as exc:
        print(f"[archive] skipped: {exc}")


def _collect():
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "consolidate_results.py")],
                       capture_output=True, text=True, timeout=600)
    except Exception:
        pass


def _dispatch(task, cfg, verify):
    meta = TASKS[task]
    if meta["family"] == "msebench_f4" and meta.get("runner") == "revision6":
        return run_revision6(task, cfg, verify)
    if meta.get("runner") == "1417":
        return run_1417(task, cfg, verify)
    if meta.get("runner") == "1920":
        return run_1920(task, cfg, verify)
    if meta["family"] == "msebench_f2":
        return run_chembench(task, cfg, verify)
    if meta["family"] == "msebench_f1":
        return run_harbor(task, cfg, verify)
    if meta["family"] == "msebench_f3":
        return run_organizer(task, cfg, verify)
    raise SystemExit(f"unsupported task: {task}")


def _task_failed(result):
    return (
        "error" in result
        or result.get("verify") is False
        or ("exit" in result and result["exit"] != 0)
        or bool((result.get("grader") or {}).get("error"))
    )


def _run_task(task, cfg, verify):
    try:
        previous = set() if verify else set(_task_artifact_dirs(task, TASKS[task]["family"]))
        result = _dispatch(task, cfg, verify)
        if not verify and not _task_failed(result):
            _stamp_model(task, cfg, previous)
        return result
    except (Exception, SystemExit) as exc:
        return {"task": task, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="comma list, 'all', or 'textqa'")
    parser.add_argument("--profile", default=None, help="llm profile name in configs/llm_config.json")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--input", default=None, help="TextQA JSONL path")
    parser.add_argument("--verify", action="store_true", help="startup-only checks")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    if args.task == "textqa" and args.verify:
        from msebench.textqa import read_items

        dataset = json.loads((HARNESS / "textqa/task.json").read_text(encoding="utf-8"))["dataset"]
        path = ROOT / (args.input or dataset["path"])
        try:
            counts = {language: len(read_items(path, language)) for language in dataset["languages"]}
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({"task": "textqa", "verify": True, "rows": counts}))
        return 0

    cfg = load_profile(args.profile)
    cfg["_profile_name"] = args.profile or cfg.get("profile") or ""

    if args.task == "textqa":
        return run_textqa(cfg, args.concurrency, args.input)

    all_order = [str(n) for n in range(1, 21)]
    tasks = all_order if args.task == "all" else [t.strip() for t in args.task.split(",") if t.strip()]
    tasks = list(dict.fromkeys(t for t in tasks if t and t != "textqa"))
    if not tasks:
        parser.error("at least one task is required")
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task: {t}")

    # 2026-09-27 fix: verify mode is offline (install/preflight/smoke/mock) — skip the gateway probe.
    if not args.verify and not _gateway_ready(cfg):
        raise SystemExit(
            "LLM gateway unreachable after retries; aborting BEFORE container "
            "time is burned (2026-09-22: an overnight gateway window cost a "
            "whole --task all batch). Retry the batch when the endpoint is up.")

    results = {}
    if args.concurrency > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_run_task, t, cfg, args.verify): t for t in tasks}
            for future in as_completed(futures):
                t = futures[future]
                results[t] = future.result()
                _collect()
    else:
        for t in tasks:
            results[t] = _run_task(t, cfg, args.verify)
            _collect()

    out = ensure_dir(RESULTS_ROOT / "run_summary") / f"{timestamp()}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _archive_runs()
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return int(any(_task_failed(result) for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
