#!/usr/bin/env python3
"""Organizer-side state server for hidden multi-case benchmarks.

The adapter is private and must expose ``create_adapter(snapshot: Path)``.  An adapter
implements ``case_count``, ``public_case(index)``, ``call_tool(index, name, arguments)``
and ``submit(index, candidate, tool_events)``.  Only allowlisted adapter responses cross
the boundary; hidden suite files and evaluator modules never enter the Agent container.
"""
from __future__ import annotations
import argparse,importlib.util,json,os,secrets,shutil,threading,traceback,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

PUBLIC_CASE_FIELDS={'case_id','task','constraints','budget','output','public_tools'}
PUBLIC_TOOL_FIELDS={'ok','tool','call_id','result','error','budget'}
PUBLIC_SUBMIT_FIELDS={'ok','case_id','status','feedback','cases_remaining','evidence_id'}

def _copytree_contents(source, target):
 """Copy a tree without chmod/utime metadata operations on bind mounts."""
 source, target = Path(source), Path(target)
 target.mkdir(parents=True, exist_ok=True)
 for root, directories, files in os.walk(source):
  relative=Path(root).relative_to(source); destination=target/relative
  destination.mkdir(parents=True, exist_ok=True)
  for directory in directories: (destination/directory).mkdir(exist_ok=True)
  for name in files: shutil.copyfile(Path(root)/name,destination/name)

def _load(path,snapshot):
 spec=importlib.util.spec_from_file_location('chembench_private_suite_adapter',path)
 if spec is None or spec.loader is None: raise RuntimeError('cannot load adapter')
 mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
 return mod.create_adapter(snapshot)

def _allow(value,fields):
 if not isinstance(value,dict): raise TypeError('adapter response must be an object')
 extra=set(value)-fields
 if extra: raise ValueError('private adapter returned undeclared fields')
 return value

class State:
 def __init__(self,adapter,evidence_dir=None): self.adapter=adapter; self.evidence_dir=evidence_dir; self.sessions={}; self.lock=threading.Lock()
 def start(self):
  token=secrets.token_urlsafe(24)
  adapter_state=self.adapter.start_session(self.evidence_dir,token) if hasattr(self.adapter,'start_session') else None
  with self.lock: self.sessions[token]={'index':0,'calls':0,'events':[],'submissions':[],'adapter_state':adapter_state}
  return token
 def get(self,token):
  with self.lock:
   if token not in self.sessions: raise KeyError('unknown_session')
   return self.sessions[token]
 def export_workspace(self,token,label='last_submission'):
  """Atomically snapshot the sidecar workspace into organizer evidence.

  The agent container receives this volume at /organizer_evidence.  A snapshot
  is made *before* adapter.submit so an adapter error cannot strand completed
  model artifacts in the private sidecar filesystem.
  """
  if not self.evidence_dir: return None
  session=self.get(token); adapter_state=session.get('adapter_state') or {}
  root=adapter_state.get('root') if isinstance(adapter_state,dict) else None
  if not root: return None
  root=Path(root)
  if not root.is_dir(): raise FileNotFoundError('sidecar_workspace_missing')
  parent=Path(self.evidence_dir)/token; parent.mkdir(parents=True,exist_ok=True)
  final=parent/label; staging=parent/(f'.{label}.{uuid.uuid4().hex}.tmp')
  try:
   requested=self.adapter.export_paths(adapter_state) if hasattr(self.adapter,'export_paths') else None
   workspace=staging/'workspace'; workspace.mkdir(parents=True)
   if requested is None:
    # Evidence may be a Docker Desktop bind mount.  `copy2` tries to restore
    # timestamps/metadata, which is rejected by some DrvFS mount setups after
    # the bytes have already been copied.  The verifier needs file contents,
    # not source metadata.
    _copytree_contents(root,workspace)
   else:
    for raw in requested:
     rel=Path(raw)
     if rel.is_absolute() or '..' in rel.parts: continue
     source=root/rel
     if not source.exists(): continue
     target=workspace/rel; target.parent.mkdir(parents=True,exist_ok=True)
     if source.is_dir(): _copytree_contents(source,target)
     else: shutil.copyfile(source,target)
   if final.exists(): shutil.rmtree(final)
   os.replace(staging,final)
  except Exception:
   shutil.rmtree(staging,ignore_errors=True)
   raise
  return f'{token}/{label}'

def handler(state):
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*_): pass
  def reply(self,status,value):
   data=json.dumps(value,separators=(',',':')).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
  def body(self):
   size=int(self.headers.get('Content-Length','0'))
   if size<0 or size>1_048_576: raise ValueError('invalid_body_size')
   value=json.loads(self.rfile.read(size) or b'{}')
   if not isinstance(value,dict): raise ValueError('body_must_be_object')
   return value
  def do_GET(self):
   self.reply(200,{'ok':True}) if self.path=='/health' else self.reply(404,{'ok':False,'error':'not_found'})
  def do_POST(self):
   try:
    body=self.body()
    if self.path=='/session':
     token=state.start(); session=state.get(token); case=_allow(state.adapter.public_case(0),PUBLIC_CASE_FIELDS)
     self.reply(200,{'ok':True,'session_id':token,'case_index':1,'case_count':state.adapter.case_count,'case':case}); return
    session=state.get(body.get('session_id')); index=session['index']
    if index>=state.adapter.case_count: raise ValueError('suite_complete')
    if self.path=='/tool':
     # Calls are consumed before validation/execution, so malformed calls cannot probe for free.
     session['calls']+=1
     public_budget=state.adapter.public_case(index).get('budget',{})
     raw_budget=public_budget.get('total_tool_calls',public_budget.get('max_tool_calls',0))
     # ``None``/``unlimited`` is the public contract for an action-unbounded
     # suite.  Do not coerce it with int(), because int(None) aborts every
     # otherwise valid tool request before the adapter can execute it.
     budget=None if raw_budget in (None,'unlimited') else int(raw_budget)
     if budget is not None and session['calls']>budget: raise ValueError('tool_quota_exceeded')
     result=_allow(state.adapter.call_tool(index,body.get('tool'),body.get('arguments'),session.get('adapter_state')),PUBLIC_TOOL_FIELDS)
     session['events'].append({'tool':body.get('tool'),'arguments':body.get('arguments'),'response':result})
     self.reply(200,result); return
    if self.path=='/submit':
     # Do not let an adapter-side copy/evaluation failure lose the model's
     # workspace.  The agent can import this fallback snapshot atomically.
     export_id=state.export_workspace(body.get('session_id'))
     result=_allow(state.adapter.submit(index,body.get('candidate'),list(session['events']),session.get('adapter_state')),PUBLIC_SUBMIT_FIELDS)
     if export_id:
      result=dict(result); result['export_id']=export_id
     session['submissions'].append(result); session['index']+=1; session['calls']=0; session['events']=[]
     if session['index']<state.adapter.case_count:
      result=dict(result); result['next_case']=_allow(state.adapter.public_case(session['index']),PUBLIC_CASE_FIELDS)
     self.reply(200,result); return
    self.reply(404,{'ok':False,'error':'not_found'})
   except (KeyError,ValueError,TypeError,json.JSONDecodeError) as exc: self.reply(400,{'ok':False,'error':type(exc).__name__})
   except Exception as exc:
    # Previously an OSError in adapter.submit escaped BaseHTTPRequestHandler,
    # which closed the TCP stream and appeared to callers as a remote-end
    # disconnect.  Return a bounded error object and retain traceback in
    # container stderr for organizer diagnosis.
    traceback.print_exc()
    self.reply(500,{'ok':False,'error':'sidecar_internal_error','error_type':type(exc).__name__})
 return Handler

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--snapshot',type=Path,required=True); ap.add_argument('--evidence-dir',type=Path); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,required=True); args=ap.parse_args()
 server=ThreadingHTTPServer((args.host,args.port),handler(State(_load(args.adapter,args.snapshot),args.evidence_dir)))
 try: server.serve_forever()
 finally: server.server_close()
if __name__=='__main__': main()
