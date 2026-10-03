"""Declarative semantic identities and symmetric deterministic compilation."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import unicodedata
from zoneinfo import ZoneInfo


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identifier(value):
    if not isinstance(value,str) or not value.strip():
        raise ValueError('Nonempty semantic identifier required')
    return unicodedata.normalize('NFC',value.strip()).casefold()


UNITS={
 'C':('temperature','K',1,273.15),'F':('temperature','K',5/9,255.3722222222222),'K':('temperature','K',1,0),
 'day':('duration','s',86400,0),'h':('duration','s',3600,0),'min':('duration','s',60,0),'s':('duration','s',1,0),
 'percent':('percentage_points',None,1,0),'B':('bytes','B',1,0),'KB':('bytes','B',1000,0),'MB':('bytes','B',1000000,0)}
UNIT_ALIASES={'%':'percent','days':'day','hours':'h','hour':'h','minutes':'min','seconds':'s','°C':'C',
              'celsius':'C','degrees Celsius':'C','fahrenheit':'F','kelvin':'K'}


class Registry:
    def __init__(self,config):
        if not isinstance(config,dict) or config.get('provenance','config') not in {'config','experience'}:
            raise ValueError('Registry requires explicit configuration provenance')
        self.raw=deepcopy(config)
        self.config=deepcopy(config)
        self.config.setdefault('version','canonical-v1')
        self.config.setdefault('roles',[])
        self.config.setdefault('relations',[])
        self.aliases={}
        for key in ['roles','relations']:
            seen=set()
            for declaration in self.config[key]:
                name=identifier(declaration['id'])
                if name in seen or not isinstance(declaration.get('meaning'),str) or not declaration['meaning'].strip():
                    raise ValueError('Missing/duplicate semantic declaration')
                seen.add(name)
                declaration['id']=name
                if key=='roles' and not declaration.get('entity_kind'):
                    raise ValueError('Role entity kind required')
                if key=='relations':
                    kind=declaration.get('value_type')
                    if kind not in {'boolean','number','string/entity','datetime','set/list'}:
                        raise ValueError('Unsupported relation type')
                    if kind!='number' and declaration.get('unit') is not None:
                        raise ValueError('Non-numeric relation cannot carry unit')
                    if kind=='number' and declaration.get('unit') is not None:
                        self.number(0,declaration['unit'],declaration.get('dimension'))
                for alias in [name]+declaration.get('aliases',[]):
                    alias=identifier(alias)
                    if (key,alias) in self.aliases and self.aliases[key,alias]!=name:
                        raise ValueError('Semantic alias collision')
                    self.aliases[key,alias]=name
        self.hash=digest(self.config)

    def resolve(self,key,value):
        name=identifier(value)
        if (key,name) not in self.aliases:
            raise ValueError('Undeclared semantic identifier')
        return self.aliases[key,name]

    def relation(self,name):
        name=self.resolve('relations',name)
        return next(r for r in self.config['relations'] if r['id']==name)

    def entity(self,value):
        name=identifier(value)
        matches=set()
        for entity in self.config.get('entities',[]):
            if name in [identifier(v) for v in [entity['id']]+entity.get('aliases',[])]:
                matches.add(entity['id'])
        if len(matches)>1:
            raise ValueError('Entity identity collision')
        return identifier(next(iter(matches))) if matches else name

    def number(self,value,unit=None,dimension=None):
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('Finite strict numeric value required')
        if unit is None:
            return value,None
        unit=UNIT_ALIASES.get(unit,unit)
        custom=self.config.get('units',{}).get(unit) if hasattr(self,'config') else None
        if custom:
            domain,target,scale,offset=custom['dimension'],custom['canonical_unit'],custom['scale'],custom.get('offset',0)
        elif unit in UNITS:
            domain,target,scale,offset=UNITS[unit]
        else:
            raise ValueError('Unsupported unit')
        if dimension is not None and dimension!=domain:
            raise ValueError('Incompatible dimension')
        result=value*scale+offset
        if not math.isfinite(result):
            raise ValueError('Nonfinite conversion')
        return result,target

    def value(self,value,kind,unit=None,dimension=None):
        if kind=='number': return self.number(value,unit,dimension)
        if unit is not None: raise ValueError('Non-numeric unit')
        if kind=='boolean':
            if type(value) is not bool: raise ValueError('Strict boolean required')
            return value,None
        if kind=='string/entity': return self.entity(value),None
        if kind=='set/list':
            if not isinstance(value,list): raise ValueError('Set/list required')
            return sorted({self.entity(v) for v in value}),None
        if kind=='datetime':
            parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
            if parsed.tzinfo is None:
                zone=self.config.get('datetime',{}).get('timezone')
                if not zone: raise ValueError('Ambiguous timezone')
                tz=ZoneInfo(zone)
                variants=[parsed.replace(tzinfo=tz,fold=f) for f in [0,1]]
                if variants[0].utcoffset()!=variants[1].utcoffset(): raise ValueError('DST ambiguity')
                candidate=variants[0]
                if candidate.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None)!=parsed:
                    raise ValueError('Nonexistent local time')
                parsed=candidate
            return parsed.astimezone(timezone.utc).isoformat(),None
        raise ValueError('Unsupported fact type')

    def fact(self,raw):
        f=deepcopy(raw)
        f['subject']=self.entity(f['subject'])
        f['fact_id']=identifier(f['fact_id'])
        if identifier(f['relation'])=='role':
            f['relation']='role'
            field='value' if 'value' in f else 'object'
            f[field]=self.resolve('roles',f[field])
            if f['value_type']!='string/entity' or f.get('unit') is not None: raise ValueError('Invalid role assertion')
            return f
        relation=self.relation(f['relation'])
        f['relation']=relation['id']
        if f['value_type']!=relation['value_type']: raise ValueError('Relation value type mismatch')
        field='value' if 'value' in f else 'object'
        if f['value_type']=='number' and relation.get('unit') is not None and f.get('unit') is None:
            raise ValueError('Required source unit absent')
        f[field],unit=self.value(f[field],f['value_type'],f.get('unit'),relation.get('dimension'))
        f.pop('unit',None)
        if unit is not None: f['unit']=unit
        f.setdefault('provenance',{})['canonical_registry_sha256']=self.hash
        return f

    def references(self,value):
        if not isinstance(value,dict): return deepcopy(value)
        if set(value)=={'role_ref'}: return {'role_ref':self.resolve('roles',value['role_ref'])}
        if set(value)=={'fact_ref'}:
            ref=value['fact_ref']
            if isinstance(ref,str): return {'fact_ref':identifier(ref)}
            return {'fact_ref':{'subject':self.subject(ref['subject']),'relation':self.resolve('relations',ref['relation'])}}
        return deepcopy(value)

    def subject(self,value):
        return self.references(value) if isinstance(value,dict) else self.entity(value)

    def compile(self,experience):
        exp=deepcopy(experience)
        for p in exp.get('preconditions',[]):
            if 'operator' not in p: continue
            p['subject']=self.subject(p['subject'])
            p['relation']=self.resolve('relations',p['relation'])
            field='expected_value' if 'expected_value' in p else 'expected_object'
            raw=p[field]
            literal=raw.get('literal') if isinstance(raw,dict) and set(raw)=={'literal'} else raw
            if isinstance(raw,dict) and set(raw)!={'literal'}:
                p[field]=self.references(raw)
            else:
                value,unit=self.value(literal,p['expected_value_type'],p.get('expected_unit'),
                                      'duration' if p['operator']=='MAX_AGE' else self.relation(p['relation']).get('dimension'))
                p[field]={'literal':value} if isinstance(raw,dict) else value
                p.pop('expected_unit',None)
                if unit is not None: p['expected_unit']=unit
            if 'reference_time' in p: p['reference_time']=self.value(p['reference_time'],'datetime')[0]
        action=exp.get('structured_action')
        if action:
            for parameter in action['parameters']:
                parameter['role']=self.resolve('roles',parameter['role'])
        return exp

    def for_experience(self,exp):
        roles={identifier(p['role']) for p in (exp.get('structured_action') or {}).get('parameters',[])}
        relations=set()
        for p in exp.get('preconditions',[]):
            if 'relation' in p: relations.add(identifier(p['relation']))
            value=p.get('expected_value',p.get('expected_object'))
            if isinstance(value,dict) and isinstance(value.get('fact_ref'),dict): relations.add(identifier(value['fact_ref']['relation']))
        config=deepcopy(self.config)
        config['roles']=[r for r in config['roles'] if r['id'] in roles]
        config['relations']=[r for r in config['relations'] if r['id'] in relations]
        return Registry(config)
