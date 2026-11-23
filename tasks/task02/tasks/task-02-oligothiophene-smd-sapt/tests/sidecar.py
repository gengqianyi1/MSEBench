#!/usr/bin/env python3
"""Image-contained organizer sidecar for the Task 02 private adapter."""
from __future__ import annotations
import argparse, importlib.util, json, os, secrets, shutil, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

CASE_FIELDS={"case_id","task","constraints","budget","output","public_tools"}
TOOL_FIELDS={"ok","tool","call_id","result","error","budget"}
SUBMIT_FIELDS={"ok","case_id","status","feedback","cases_remaining","evidence_id"}

def load_adapter(path, snapshot):
    spec=importlib.util.spec_from_file_location("chembench_private_suite_adapter", path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load adapter")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.create_adapter(snapshot)

def allow(value, fields):
    if not isinstance(value, dict): raise TypeError("adapter response must be an object")
    if set(value)-fields: raise ValueError("private adapter returned undeclared fields")
    return value

def copytree(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for root, directories, files in os.walk(source):
        destination=target/Path(root).relative_to(source); destination.mkdir(parents=True, exist_ok=True)
        for name in directories: (destination/name).mkdir(exist_ok=True)
        for name in files: shutil.copyfile(Path(root)/name, destination/name)

class State:
    def __init__(self, adapter, evidence_dir): self.adapter=adapter; self.evidence_dir=evidence_dir; self.sessions={}; self.lock=Lock()
    def start(self):
        token=secrets.token_urlsafe(24); state=self.adapter.start_session(self.evidence_dir,token) if hasattr(self.adapter,"start_session") else None
        with self.lock: self.sessions[token]={"index":0,"calls":0,"events":[],"adapter_state":state}
        return token
    def get(self, token):
        with self.lock:
            if token not in self.sessions: raise KeyError("unknown_session")
            return self.sessions[token]
    def export(self, token):
        if not self.evidence_dir: return None
        session=self.get(token); state=session.get("adapter_state") or {}; root=state.get("root") if isinstance(state,dict) else None
        if not root or not Path(root).is_dir(): return None
        parent=Path(self.evidence_dir)/token; parent.mkdir(parents=True,exist_ok=True); staging=parent/(".last_submission."+uuid.uuid4().hex+".tmp"); final=parent/"last_submission"
        try:
            workspace=staging/"workspace"; requested=self.adapter.export_paths(state) if hasattr(self.adapter,"export_paths") else None
            if requested is None: copytree(Path(root),workspace)
            else:
                workspace.mkdir(parents=True,exist_ok=True)
                for raw in requested:
                    relative=Path(raw)
                    if relative.is_absolute() or ".." in relative.parts: continue
                    source=Path(root)/relative
                    if not source.exists(): continue
                    target=workspace/relative; target.parent.mkdir(parents=True,exist_ok=True)
                    copytree(source,target) if source.is_dir() else shutil.copyfile(source,target)
            if final.exists(): shutil.rmtree(final)
            os.replace(staging,final)
        except Exception: shutil.rmtree(staging,ignore_errors=True); raise
        return str(Path(token)/"last_submission")

def handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def reply(self,status,value):
            data=json.dumps(value,separators=(",",":")).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
        def body(self):
            size=int(self.headers.get("Content-Length","0"))
            if size<0 or size>1048576: raise ValueError("invalid_body_size")
            value=json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(value,dict): raise ValueError("body_must_be_object")
            return value
        def do_GET(self): self.reply(200,{"ok":True}) if self.path=="/health" else self.reply(404,{"ok":False,"error":"not_found"})
        def do_POST(self):
            try:
                body=self.body()
                if self.path=="/session":
                    token=state.start(); self.reply(200,{"ok":True,"session_id":token,"case_index":1,"case_count":state.adapter.case_count,"case":allow(state.adapter.public_case(0),CASE_FIELDS)}); return
                session=state.get(body.get("session_id")); index=session["index"]
                if index>=state.adapter.case_count: raise ValueError("suite_complete")
                if self.path=="/tool":
                    session["calls"]+=1; budget_data=state.adapter.public_case(index).get("budget",{}); raw=budget_data.get("total_tool_calls",budget_data.get("max_tool_calls")); budget=None if raw in (None,"unlimited") else int(raw)
                    if budget is not None and session["calls"]>budget: raise ValueError("tool_quota_exceeded")
                    result=allow(state.adapter.call_tool(index,body.get("tool"),body.get("arguments"),session.get("adapter_state")),TOOL_FIELDS); session["events"].append({"tool":body.get("tool"),"arguments":body.get("arguments"),"response":result}); self.reply(200,result); return
                if self.path=="/submit":
                    export_id=state.export(body.get("session_id")); result=allow(state.adapter.submit(index,body.get("candidate"),list(session["events"]),session.get("adapter_state")),SUBMIT_FIELDS)
                    if export_id: result=dict(result); result["export_id"]=export_id
                    self.reply(200,result); return
                self.reply(404,{"ok":False,"error":"not_found"})
            except (KeyError,ValueError,TypeError,json.JSONDecodeError) as exc: self.reply(400,{"ok":False,"error":type(exc).__name__})
            except Exception as exc: traceback.print_exc(); self.reply(500,{"ok":False,"error":"sidecar_internal_error","error_type":type(exc).__name__})
    return Handler

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--adapter",type=Path,required=True); parser.add_argument("--snapshot",type=Path,required=True); parser.add_argument("--evidence-dir",type=Path); parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,required=True); args=parser.parse_args()
    server=ThreadingHTTPServer((args.host,args.port),handler(State(load_adapter(args.adapter,args.snapshot),args.evidence_dir)))
    try: server.serve_forever()
    finally: server.server_close()
if __name__=="__main__": main()
