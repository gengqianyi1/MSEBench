"""Private stateful adapter for the six-case native xTB recovery suite."""
from __future__ import annotations
import json,os,shutil,sys,tempfile
from pathlib import Path
import yaml

import datetime as _dt, time as _time


class _MonotonicClock:
    """Strictly increasing UTC timestamps: the wall clock can regress under WSL/NTP, which
    would make the trajectory hash-chain validator reject a otherwise-valid submission."""
    def __init__(self):
        self._base = _dt.datetime.now(_dt.timezone.utc)
        self._t0 = _time.monotonic()
        self._last = None
    def __call__(self):
        now = self._base + _dt.timedelta(seconds=(_time.monotonic() - self._t0))
        if self._last is not None and now <= self._last:
            now = self._last + _dt.timedelta(microseconds=1)
        self._last = now
        return now


class Adapter:
 def __init__(self,snapshot):
  self.runtime=Path(snapshot)/'private_data/runtime_package'; sys.path.insert(0,str(self.runtime))
  from tools.tool_registry import ToolRegistry
  from framework import ToolContractSet,TrajectoryRecorder,build_final_answer_envelope
  from baselines.baseline_agent import _special_event_payload
  self.Registry=ToolRegistry; self.Contracts=ToolContractSet; self.Recorder=TrajectoryRecorder; self.build_final_answer_envelope=build_final_answer_envelope; self.special_payload=_special_event_payload; suite=json.loads((self.runtime/'organizer_private/suite.json').read_text()); self.cases=[suite['public_case'],*suite['hidden_cases']]; self.case_count=len(self.cases)
  self.configs=[yaml.safe_load((self.runtime/c['config']).read_text()) for c in self.cases]
  contracts=json.loads((self.runtime/'configs/tool_contracts.json').read_text()); self.contracts=contracts
 def start_session(self,evidence_dir,token):
  temp=tempfile.TemporaryDirectory(prefix='chembench-task08-'); root=Path(temp.name); shutil.copytree(self.runtime/'configs',root/'configs',copy_function=shutil.copyfile,dirs_exist_ok=True,ignore_dangling_symlinks=True); shutil.copytree(self.runtime/'organizer_configs',root/'organizer_configs',copy_function=shutil.copyfile,dirs_exist_ok=True,ignore_dangling_symlinks=True)
  return {'temp':temp,'root':root,'registries':{},'evidence_dir':Path(evidence_dir) if evidence_dir else None,'token':token}
 def public_case(self,index):
  cfg=self.configs[index]; submit=next(x['function']['parameters'] for x in self.contracts['tools'] if x['function']['name']=='submit_final'); tools=[x for x in self.contracts['tools'] if x['function']['name']!='submit_final']
  task={'id':cfg['task']['id'],'case_id':cfg['task']['case_id'],'protocol':cfg['task']['protocol'],'job_spec':cfg['task']['job_spec'],'case_manifest':cfg['task']['case_manifest']}
  return {'case_id':self.cases[index]['case_id'],'task':task,'budget':cfg['budget'],'constraints':{'required_phases':['geometry_optimization','hessian'],'native_execution_required':True},'output':{'candidate_schema':submit},'public_tools':tools}
 def _registry(self,index,state):
  if index in state['registries']: return state['registries'][index]
  case=self.cases[index]; root=state['root']; source=self.runtime/case['workspace']; target=root/case['workspace']; target.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(source,target,copy_function=shutil.copyfile,dirs_exist_ok=True,ignore_dangling_symlinks=True)
  registry=self.Registry(root,task_config_path=case['config'],case_root=case['workspace']); cfg=self.configs[index]; recorder=self.Recorder(timestamp_fn=_MonotonicClock(),protocol=cfg['task']['protocol'],run_id=f'harbor-{state["token"]}-{index}',genesis_hash=cfg['determinism']['trajectory_genesis_hash'])
  recorder.append('run_start',{'task_id':cfg['task']['id'],'case_id':case['case_id'],'protocol':cfg['task']['protocol'],'run_type':'HARBOR_PROTOCOL','logical_model':'harbor_managed','requested_model':'harbor_managed','provider':'harbor','transport':'harbor_agent','execution_mode':'HARBOR','native_mode':'LIVE_XTB'})
  bundle={'registry':registry,'recorder':recorder,'calls':0,'successful':0,'attempts':0,'records':[]}; state['registries'][index]=bundle; return bundle
 def _dispatch(self,index,name,arguments,state):
  bundle=self._registry(index,state); registry=bundle['registry']; bundle['calls']+=1; call_id=f'harbor-{bundle["calls"]:03d}'; recorder=bundle['recorder']; recorder.append('tool_call',{'call_id':call_id,'name':name,'arguments':arguments}); result=registry.dispatch(name,arguments,call_id)
  if result.get('ok'):
   value=result.get('result'); bundle['successful']+=1; recorder.append('tool_result',{'call_id':call_id,'name':name,'ok':True,'result':value})
   if name in {'prepare_restart','run_xtb','validate_attempt'}: recorder.append({'prepare_restart':'attempt_prepared','run_xtb':'native_xtb_execution','validate_attempt':'attempt_validated'}[name],self.special_payload(name,value))
   if name=='prepare_restart': bundle['attempts']+=1
  else: recorder.append('tool_result',{'call_id':call_id,'name':name,'ok':False,'error':{'type':result.get('error',{}).get('code','TOOL_REPORTED_FAILURE'),'status':'REJECTED'}})
  bundle['records'].append({'call_id':call_id,'tool':name,'arguments':arguments,'response':result})
  recorder.append('budget_state',{'tool_calls':bundle['calls'],'successful_tool_calls':bundle['successful'],'agent_turns':0,'execution_attempts':bundle['attempts'],'provider_retries':0,'elapsed_seconds':0.0}); return result
 def call_tool(self,index,name,arguments,state):
  if name=='submit_final': return {'ok':False,'tool':name,'error':'submit_through_candidate_endpoint'}
  result=self._dispatch(index,name,arguments,state)
  if result.get('ok'): return {'ok':True,'tool':name,'call_id':result.get('call_id'),'result':result.get('result')}
  return {'ok':False,'tool':name,'call_id':result.get('call_id'),'error':result.get('error',{}).get('code','tool_error')}
 def submit(self,index,candidate,tool_events,state):
  bundle=self._registry(index,state); registry=bundle['registry']; result=self._dispatch(index,'submit_final',candidate,state); case_id=self.cases[index]['case_id']; evidence_id=f'{state["token"]}/{case_id}'
  if result.get('ok'):
   recorder=bundle['recorder']; recorder.append('final_submitted',{'final_answer':candidate}); recorder.append('run_complete',{'status':'COMPLETE','terminal_status':'COMPLETE','actual_model':None,'tool_calls':bundle['calls'],'successful_tool_calls':bundle['successful']})
  if result.get('ok') and state['evidence_dir']:
   target=state['evidence_dir']/evidence_id; target.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(state['root']/self.cases[index]['workspace'],target/'workspace',copy_function=shutil.copyfile,dirs_exist_ok=True,ignore_dangling_symlinks=True); trajectory=target/'trajectory.jsonl'; bundle['recorder'].write(trajectory)
   envelope=self.Contracts.load(self.runtime/'configs/tool_contracts.json').validate_final_answer_artifact(self.build_final_answer_envelope(model_submission=candidate,events=bundle['recorder'].events,trajectory_path=trajectory,artifact_root=target/'workspace',genesis_hash=self.configs[index]['determinism']['trajectory_genesis_hash']))
   requested_model=os.getenv('CHEMBENCH_MODEL_ID'); actual_model=os.getenv('CHEMBENCH_ACTUAL_MODEL_ID')
   metadata={'schema_version':'task08-organizer-evidence/1.0','session_id':state['token'],'case_id':case_id,'provider_status':'HARBOR_PROTOCOL','requested_model_id':requested_model,'actual_model_id':actual_model,'exact_model_identity':bool(requested_model and actual_model==requested_model)}
   (target/'final_answer.json').write_text(json.dumps(envelope,indent=2)); (target/'trusted_state.json').write_text(json.dumps(registry.trusted_state,indent=2)); (target/'tool_records.json').write_text(json.dumps(bundle['records'],indent=2)); (target/'candidate.json').write_text(json.dumps(candidate,indent=2)); (target/'metadata.json').write_text(json.dumps(metadata,indent=2))
  feedback={'accepted':result.get('ok') is True,'final_status':candidate.get('final_status') if isinstance(candidate,dict) else None}
  return {'ok':result.get('ok') is True,'case_id':case_id,'status':'submitted' if result.get('ok') else 'invalid','feedback':feedback,'cases_remaining':self.case_count-index-1,'evidence_id':evidence_id}
def create_adapter(snapshot): return Adapter(snapshot)
