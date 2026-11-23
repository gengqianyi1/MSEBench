"""Framework-owned protected tool gateway primitives."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from .state import StateStore


@dataclass(frozen=True)
class ToolContract:
    name: str
    public_fields: frozenset[str]


class ProtectedToolGateway:
    """Persist quotas before dispatch and fail closed on undeclared response fields."""
    def __init__(self, state: StateStore, handlers: Mapping[str, Callable[[dict[str,Any]],dict[str,Any]]], contracts: Mapping[str,ToolContract], max_calls: int):
        if max_calls < 0: raise ValueError('max_calls must be non-negative')
        self.state=state; self.handlers=dict(handlers); self.contracts=dict(contracts); self.max_calls=max_calls

    def execute(self, request: Any) -> dict[str,Any]:
        current=self.state.load()
        if current is None: raise RuntimeError('gateway state must be initialized')
        used=int(current.task_state.get('gateway_calls_used',0))
        if used >= self.max_calls: return self._error('quota_exceeded',used)
        current.task_state['gateway_calls_used']=used+1; self.state.save(current); used+=1
        if not isinstance(request,dict): return self._error('malformed_tool_call',used)
        name=request.get('tool'); arguments=request.get('arguments',{})
        if not isinstance(name,str) or not name: return self._error('malformed_tool_call',used)
        if name not in self.handlers or name not in self.contracts: return self._error('unknown_tool',used)
        if not isinstance(arguments,dict): return self._error('invalid_arguments',used)
        try: private_result=self.handlers[name](arguments)
        except Exception: return self._error('tool_execution_failed',used)
        if not isinstance(private_result,dict): return self._error('invalid_tool_result',used)
        if set(private_result)-set(self.contracts[name].public_fields):
            return self._error('response_contract_violation',used)
        return {'ok':True,'tool':name,'result':private_result,'calls_used':used,'calls_remaining':self.max_calls-used}

    def _error(self,code:str,used:int)->dict[str,Any]:
        return {'ok':False,'error':{'code':code},'calls_used':used,'calls_remaining':max(0,self.max_calls-used)}
