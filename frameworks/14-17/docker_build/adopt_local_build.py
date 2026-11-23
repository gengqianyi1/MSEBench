#!/usr/bin/env python3
"""Make the 14-17 package accept images that were rebuilt from scratch on this host.

WHY
---
The package's run scripts (./chembench oracle|llm|llm-mock|preflight) refuse to start unless every image
matches docker/image_manifest.json by layer digest -- the guard against stale images from an earlier
revision. A from-scratch rebuild installs the same pinned packages but necessarily produces different
layers (new timestamps, new base-image pulls), so without this step every run stops with
"local layers differ from the shipped image".

WHAT IT DOES
------------
1. checks that every image the tasks start exists locally and matches this Docker daemon's platform;
2. tags Harbor's egress-control image under the name Harbor 0.21.0 looks up on this platform
   (the compose file builds it under the linux/amd64 name);
3. saves the shipped manifest once as docker/image_manifest.shipped.json and rewrites
   docker/image_manifest.json so its images[] describe the local build;
4. runs the package's own image check (scripts/offline_images.py check --all-roles).

Undo:  mv docker/image_manifest.shipped.json docker/image_manifest.json   (then ./chembench load-images)

Runs inside the runner image (needs harbor importable and the Docker CLI); build.sh does that:
    python adopt_local_build.py <package dir>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EGRESS_BUILT = "harbor-prebuilt:harbor-docker-egress-control-sidecar--b1e6760734a9e6f6"  # compose tag


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    pkg = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(pkg / "scripts"))
    import offline_images as oi  # the package's own module: same image list, same identity rule

    platform = oi.daemon_platform()
    if platform is None:
        print("! cannot reach a Docker daemon")
        return 1

    egress = oi.egress_image_name(platform)
    if oi.inspect(egress) is None:
        if oi.inspect(EGRESS_BUILT) is None:
            print(f"! egress image missing: build service harbor-egress-control first ({EGRESS_BUILT})")
            return 1
        subprocess.run(["docker", "tag", EGRESS_BUILT, egress], check=True)
        print(f"  tagged {EGRESS_BUILT} -> {egress}  (the name Harbor looks up on {platform})")

    wanted = [(r["role"], r["task"], r["reference"], r["recipe"]) for r in oi.required()]
    wanted += [("egress", None, egress, {"harbor_builtin": "harbor/environments/docker/"
                                                            "harbor-docker-egress-control-sidecar"}),
               ("runner", None, oi.runner_image(), {"dockerfile": "docker/runner/Dockerfile",
                                                    "context": "requirements.txt"})]
    entries, problems = [], []
    for role, task, ref, recipe in wanted:
        info = oi.inspect(ref)
        if info is None:
            problems.append(f"MISSING {role} {ref}")
        elif info["platform"] != platform:
            problems.append(f"{role} {ref} is {info['platform']} but this Docker daemon is {platform}; "
                            f"rebuild with CHEMBENCH_PLATFORM={platform}")
        else:
            entries.append({"role": role, "task": task, "reference": ref, **info, "recipe": recipe})
    if problems:
        for p in problems:
            print(f"! {p}")
        return 1

    manifest = oi.load_manifest() or {}
    backup = oi.MANIFEST.with_name("image_manifest.shipped.json")
    if not backup.exists() and oi.MANIFEST.is_file():
        shutil.copy2(oi.MANIFEST, backup)
        print(f"  saved shipped manifest -> docker/{backup.name}")
    manifest.update({
        "platform": platform,
        "harbor_version": oi.harbor_version(),
        "images": entries,
        "local_rebuild": {
            "rebuilt_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "how": "14_17_revision10_docker_build/docker-compose.build.yaml (from-scratch build)",
            "note": "images[] describe the locally rebuilt images. archive* fields still describe the shipped "
                    "offline archive; loading that archive again requires restoring "
                    "docker/image_manifest.shipped.json first.",
        },
    })
    oi.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  docker/{oi.MANIFEST.name}: {len(entries)} locally built images recorded\n")
    return oi.cmd_check(None, False, True, False)


if __name__ == "__main__":
    sys.exit(main())
