"""Model-backed extraction with independent evidence request; no decision policy."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import re
from jsonschema import Draft202012Validator
from canonical_declarations import Registry, digest, identifier
from canonical_wire import MODEL, EXTRACTION_PROMPT, VERIFICATION_PROMPT, extraction_schema, verification_schema, request, content, assemble, verification_payload, attach_verified, semantic_key


SETTINGS={'requested_model':MODEL,'substitution_allowed':False,'temperature':0,'top_p':1,'max_tokens':4096,
          'timeout_seconds':45,'retry_count':0,'enable_thinking':True}


def validate_fact(f,registry):
    required={'fact_id','subject','relation','value_type','source','evidence','confidence'}
    allowed=required|{'value','object','unit','provenance'}
    if not isinstance(f,dict) or not required<=set(f) or set(f)-allowed or ('value' in f)==('object' in f):
        raise ValueError('Full Fact contract required')
    if not all(isinstance(f[k],str) and f[k].strip() for k in ['fact_id','subject','relation','evidence']):
        raise ValueError('Fact identity/evidence required')
    if not isinstance(f['source'],dict): raise ValueError('Source identity required')
    c=f['confidence']
    if c is not None and (type(c) not in (int,float) or not 0<=c<=1): raise ValueError('Invalid confidence')
    registry.fact(f)
    return deepcopy(f)


def validate_source(f,text,registry,context=None):
    validate_fact(f,registry)
    if f['source'].get('kind')=='data':
        context=context or []
        i=f['source'].get('index')
        if type(i) is not int or not 0<=i<len(context) or f['source'].get('input_sha256')!=digest(context) or f['evidence']!='/data/'+str(i):
            raise ValueError('Invalid structured source pointer')
        datum=context[i]
        for key in ['subject','relation','value_type']:
            if identifier(f[key])!=identifier(datum[key]): raise ValueError('Structured source identity mismatch')
        if f.get('value',f.get('object'))!=datum.get('value',datum.get('object')) or f.get('unit')!=datum.get('unit'):
            raise ValueError('Structured source value/unit mismatch')
        return deepcopy(f)
    if f['source']!={'kind':'text','input_sha256':hashlib.sha256(text.encode()).hexdigest()}:
        raise ValueError('Source identity mismatch')
    quote=f['evidence']
    starts=[m.start() for m in re.finditer(re.escape(quote),text)]
    prov=f.get('provenance',{})
    span=prov.get('span')
    if span is None:
        if len(starts)!=1: raise ValueError('Ambiguous/missing source occurrence')
        span={'start':starts[0],'end':starts[0]+len(quote)}
    if not isinstance(span,dict) or set(span)!={'start','end'} or type(span['start']) is not int or type(span['end']) is not int:
        raise ValueError('Invalid source span')
    if not 0<=span['start']<span['end']<=len(text) or text[span['start']:span['end']]!=quote:
        raise ValueError('Source span mismatch')
    relation=None if identifier(f['relation'])=='role' else registry.relation(f['relation'])
    global_scope=relation and relation.get('subject_scope')=='global' and identifier(f['subject'])==identifier(relation['scope_entity'])
    if global_scope:
        if not prov.get('mention') or prov['mention'] not in quote: raise ValueError('Global declaration mention missing')
    elif prov.get('mention')!=f['subject'] or not re.search(r'(?<!\w)'+re.escape(f['subject'])+r'(?!\w)',quote):
        raise ValueError('Entity mention not supported')
    return {**deepcopy(f),'provenance':{**prov,'span':span}}


def verify_claim(f,result,text):
    flags={'entity','relation','value','unit','polarity','source'}
    if (not isinstance(result,dict) or result.get('fact_id')!=f['fact_id'] or result.get('state')!='SUPPORTED'
        or set(result.get('checks',{}))!=flags or any(v is not True for v in result['checks'].values())
        or result.get('evidence')!=f['evidence'] or
        (f['source'].get('kind')!='data' and result['evidence'] not in text)):
        return False
    return True


class CanonicalExtractor:
    mode='canonical'
    def __init__(self,registry,transport=None,model_id=MODEL,technical_params=None):
        self.registry=registry if isinstance(registry,Registry) else Registry(registry)
        self.active=self.registry
        self.transport=transport
        self.model_id=model_id
        self.technical_params=deepcopy(technical_params)
        self.model_calls=0
        self.last_trace=None
        self.declarations={}

    def prepare_experience(self,exp):
        self.active=self.registry.for_experience(exp)
        self.compile_error=None
        try: compiled=self.active.compile(exp)
        except (ValueError,KeyError,TypeError) as exc:
            compiled=deepcopy(exp)
            self.compile_error=type(exc).__name__
        self.declarations={'raw':deepcopy(exp),'compiled':deepcopy(compiled),
                           'original_sha256':digest(exp),'compiled_sha256':digest(compiled),
                           'registry_sha256':self.active.hash,'compile_error':self.compile_error}
        return compiled

    def _call(self,purpose,req,trace):
        counts=trace['call_counters']
        counts[purpose+'_attempts']+=1
        self.model_calls+=1
        metadata={'purpose':purpose,'endpoint':'https://api.tokenfactory.nebius.com/v1/chat/completions',
                    'requested_model':req['model'],'sampling':{k:req[k] for k in ['temperature','top_p','max_tokens']},
                  'prompt_sha256':digest(req['messages'][0]['content']),
                  'schema_sha256':digest(req['response_format']['json_schema']['schema']),
                  'request_sha256':digest(req),'started_at':datetime.now(timezone.utc).isoformat()}
        trace['requests'].append(metadata)
        try:
            if self.transport is None:
                from model_client import NebiusClient
                body,_=NebiusClient()._request('/chat/completions',req)
            else: body=self.transport(purpose,req)
            metadata['returned_model']=body.get('model')
            metadata['usage']=body.get('usage')
            metadata['thinking']=deepcopy(req.get('chat_template_kwargs'))
            metadata['finish_reason']=(body.get('choices') or [{}])[0].get('finish_reason')
            if body.get('model')!=req['model']: raise ValueError('Model substitution')
            result=content(body,req['response_format']['json_schema']['schema'])
            counts[purpose+'_successes']+=1
            return result
        except Exception as exc:
            counts[purpose+'_errors']+=1
            metadata['error']=type(exc).__name__
            raise
        finally: metadata['completed_at']=datetime.now(timezone.utc).isoformat()

    def extract(self,text,context=None):
        trace={'facts':[],'rejected_facts':[],'errors':[],'status':'UNKNOWN','extraction_model_calls':0,
               'raw_candidate_facts':[],'verification_results':[],'requests':[],
               'call_counters':{k:0 for p in ['extraction','verification'] for k in [p+'_attempts',p+'_successes',p+'_errors']},
               'declarations':deepcopy(self.declarations),'registry_sha256':self.active.hash,
               'registry':deepcopy(self.active.config)}
        self.last_trace=trace
        schema=extraction_schema(self.active.config,text_input=text,context=context)
        payload={'source':{'text':text},'registry':self.active.config}
        if context:
            payload['source']['data']=deepcopy(context)
        try:
            raw=self._call('extraction',self._request(EXTRACTION_PROMPT,payload,schema,'canonical_facts'),trace)
            trace['internal_model_output']=deepcopy(raw)
            if raw['status']!='OK': raise ValueError('Extraction status UNKNOWN')
            claims=assemble(raw,text,self.active.config,context)
            trace['raw_candidate_facts']=deepcopy(claims)
            if len(claims)>64: raise ValueError('Claim count exceeded')
            ids=[identifier(f['fact_id']) for f in claims]
            if len(set(ids))!=len(ids): raise ValueError('Duplicate canonical fact IDs')
            self.verify_candidates(text,claims,context,trace)
            if getattr(self,'compile_error',None): raise ValueError('Declaration compilation failed')
            trace['status']='OK' if not trace['errors'] else 'UNKNOWN'
        except Exception as exc:
            trace['errors'].append('Semantic stage failure: '+type(exc).__name__)
            for claim in trace['raw_candidate_facts']:
                if not any(r['fact'].get('fact_id')==claim.get('fact_id') for r in trace['rejected_facts']):
                    self.reject(trace,claim,type(exc).__name__)
            trace['facts']=[]
        trace['extraction_model_calls']=sum(trace['call_counters'][p+'_attempts'] for p in ['extraction','verification'])
        return deepcopy(trace)

    def _request(self,prompt,payload,schema,name):
        return request(prompt,payload,schema,name,self.model_id,self.technical_params)

    def verify_candidates(self,text,claims,context=None,trace=None):
        """Production verifier-only entry; one batch after ordinary validation."""
        if trace is None:
            trace={'facts':[],'rejected_facts':[],'errors':[],'verification_results':[], 'requests':[],
                   'raw_candidate_facts':deepcopy(claims),
                   'call_counters':{k:0 for p in ['extraction','verification'] for k in [p+'_attempts',p+'_successes',p+'_errors']}}
        self.last_trace=trace
        valid=[]
        for claim in claims:
            try: valid.append(validate_source(claim,text,self.active,context))
            except (ValueError,KeyError,TypeError) as exc:self.reject(trace,claim,type(exc).__name__)
        try:
            if len(claims)>64 or len({c['fact_id'] for c in claims})!=len(claims):raise ValueError('Claim identities/count')
            if valid:
                schema=verification_schema(valid)
                output=self._call('verification',self._request(VERIFICATION_PROMPT,verification_payload(valid,self.active.config,context),schema,'canonical_verification'),trace)
                rows=attach_verified(output['verifications'],valid)
                trace['verification_results']=deepcopy(rows)
                for claim in valid:
                    result=next(r for r in rows if r['fact_id']==claim['fact_id'])
                    if verify_claim(claim,result,text):trace['facts'].append(self.active.fact(claim))
                    else:self.reject(trace,claim,'Verification '+result['state'])
        except Exception as exc:
            trace['errors'].append('Verification failure: '+type(exc).__name__)
            for claim in valid:
                if not any(r['fact'].get('fact_id')==claim.get('fact_id') for r in trace['rejected_facts']):self.reject(trace,claim,type(exc).__name__)
            trace['facts']=[]
        accepted=deepcopy(trace['facts']); grouped={}
        outcomes={r['fact_id']:r for r in trace['verification_results']}
        rejected={r['fact'].get('fact_id'):r for r in trace['rejected_facts']}
        for claim in claims:
            try:key=semantic_key(claim,self.active)
            except (ValueError,KeyError,TypeError):continue
            grouped.setdefault(key,[]).append({'candidate':deepcopy(claim),'verification':outcomes.get(claim['fact_id']),
                                             'rejection':rejected.get(claim['fact_id'])})
        trace['semantic_groups']=[{'key':key,'records':records,'state':'UNKNOWN' if any(r['rejection'] for r in records) else 'CONFIRMED',
                                  'duplicate_count':len(records)-1} for key,records in grouped.items()]
        trace['facts']=[]; seen=set()
        for fact in accepted:
            key=semantic_key(fact,self.active)
            if key in seen or any(r['rejection'] for r in grouped.get(key,[])):continue
            seen.add(key)
            fact['provenance']['evidence_records']=deepcopy(grouped[key])
            trace['facts'].append(fact)
        return deepcopy(trace)

    def reject(self,trace,claim,reason):
        f=deepcopy(claim)
        # Rejections need canonical identities so the unchanged evaluator can
        # block exactly the matching required relation rather than all conditions.
        try:
            f['subject']=self.active.entity(f['subject'])
            f['relation']='role' if identifier(f['relation'])=='role' else self.active.resolve('relations',f['relation'])
            if f['relation']=='role':
                field='value' if 'value' in f else 'object'
                f[field]=self.active.resolve('roles',f[field])
        except (ValueError,KeyError,TypeError): pass
        trace['rejected_facts'].append({'fact':f,'reason':reason,'evaluation':'UNKNOWN'})
