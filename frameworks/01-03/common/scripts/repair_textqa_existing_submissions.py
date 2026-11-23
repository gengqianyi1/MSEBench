#!/usr/bin/env python3
"""Grade previously saved TextQA responses after a parser-only recovery."""
from __future__ import annotations
import argparse, json, statistics, tempfile, shutil
from pathlib import Path
from textqa_deepseek_smoke import ROOT, extract_submission
from textqa_shared_verifier import SharedTextQAVerifier, make_private_index

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("run_dir", type=Path); args=p.parse_args()
    run=args.run_dir.resolve(); data=json.loads((run/"results.json").read_text(encoding="utf-8"))
    dataset=ROOT/"textqa"/"generated"/"textqa_v001"; manifest=json.loads((dataset/"manifest.json").read_text(encoding="utf-8"))
    by_name={x["task_name"]:x for x in manifest["tasks"]}
    private=Path(tempfile.mkdtemp(prefix="textqa-private-index-")); verifier=SharedTextQAVerifier(make_private_index(dataset, private/"reference_index.json"))
    try:
        for row in data["results"]:
            if row["score"] is not None: continue
            item=by_name[row["task"]]; item_dir=next(run.glob(f"{row['id']}_*"))
            response=json.loads((item_dir/"raw_response.json").read_text(encoding="utf-8"))
            submission=extract_submission(response)
            write(item_dir/"workspace/final_answer.json", submission)
            graded=verifier.grade(row["task"], submission["answer"])
            v=item_dir/"verifier_logs/verifier"
            details={"schema_version":"textqa-verifier-details/1.0","status":graded["status"],"task_id":row["task"],"private_result":{"score":graded["score"],"subscores":{},"failure_codes":graded["failure_codes"],"evidence":{"question_id":item["position"],"predicted_answer":submission["answer"]}}}
            write(v/"details.json",details); write(v/"grader-result.json",{"schema_version":"chembench-grader-result/1.0","reward":graded["reward"],"raw_score":graded["score"],"status":graded["status"]}); write(v/"reward.json",{"reward":graded["reward"]})
            row.update({"submission_valid":True,"grader_status":graded["status"],"score":graded["score"],"reward":graded["reward"],"error":None,"parser_recovery":"leading answer field recovered from provider-truncated JSON; no model rerun"})
            write(item_dir/"summary.json",row)
    finally:
        verifier.close(); shutil.rmtree(private)
    scores=[float(x["score"]) for x in data["results"] if isinstance(x["score"],(int,float))]
    a=data["aggregate"]; a.update({"successfully_evaluated":len(scores),"failed_evaluation":len(data["results"])-len(scores),"mean_score":statistics.mean(scores),"median_score":statistics.median(scores),"zero_score_count":sum(x==0 for x in scores),"nonzero_count":sum(x!=0 for x in scores),"accuracy":sum(x==100 for x in scores)/len(scores)})
    write(run/"results.json",data); print(json.dumps(a,ensure_ascii=False))
if __name__=="__main__": main()
