"""Lossless, indexed G2 acceleration. Trusted local artifacts only, like G1.

No normalization, candidate enumeration, tensorization, model or sampling here.
SQLite indexes immutable logical identities. Compressed templates contain only
byte-identical stored content; per-sample records keep all varying fields.
Normal open never scans all records. Reads authenticate just the requested
record/template. A bounded template LRU is private; returned tensors are owned
by the caller. Semantic identity is independent of pickle/set byte ordering;
record checksums authenticate actual bytes separately.
"""
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import is_dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import zlib

import torch

VERSION='mssr.bc_v1.g2_sidecar.v1'
IDENTITIES=('canonical_sha','f1_sha','g1_fingerprint','f2_fingerprint','normalizer_sha',
    'layout_fingerprint','feature_schema_fingerprint','robot_spec_fingerprint',
    'tensorizer_fingerprint','storage_fingerprint','split_fingerprint')
SPLITS=('train','validation','test')
_DYNAMIC_ROUTING=frozenset(('audit','resource_targets','structural_targets','physical_supervision',
    'environment_refs','environment_layouts','object_refs'))
_STATIC_INPUTS=frozenset(('structural','resource_candidates'))


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _hash(value):return hashlib.sha256(_json(value).encode()).hexdigest()
def _sha(blob):return hashlib.sha256(blob).hexdigest()


def _tensor(dtype,shape,raw):
    if dtype not in ('float32','int64','bool'):raise ValueError('Unsupported G2 tensor dtype')
    d=getattr(torch,dtype)
    return torch.frombuffer(bytearray(raw),dtype=d).reshape(shape) if raw else torch.empty(shape,dtype=d)


class _Pickler(pickle.Pickler):
    def reducer_override(self,value):
        if isinstance(value,torch.Tensor):
            if value.device.type!='cpu':raise ValueError('Sidecar stores CPU G2 tensors only')
            return _tensor,(str(value.dtype).split('.')[-1],tuple(value.shape),value.detach().contiguous().numpy().tobytes())
        return NotImplemented


def encode(value):
    stream=BytesIO();_Pickler(stream,protocol=5).dump(value)
    return zlib.compress(stream.getvalue(),1)


def decode(blob):
    try:return pickle.loads(zlib.decompress(blob))
    except Exception as exc:raise ValueError('Corrupt sidecar record') from exc


def assert_exact(actual,expected,path='sample'):
    """Storage verification only; dataclasses are the unchanged typed routing."""
    if isinstance(expected,torch.Tensor):
        equal=isinstance(actual,torch.Tensor) and actual.dtype==expected.dtype and actual.shape==expected.shape and torch.equal(actual,expected)
    elif isinstance(expected,dict):
        equal=isinstance(actual,dict) and actual.keys()==expected.keys()
        if equal:
            for k in expected:assert_exact(actual[k],expected[k],path+'.'+str(k))
    elif isinstance(expected,(tuple,list)):
        equal=type(actual) is type(expected) and len(actual)==len(expected)
        if equal:
            for i,(a,b) in enumerate(zip(actual,expected)):assert_exact(a,b,path+'.'+str(i))
    else:equal=actual==expected
    if not equal:raise ValueError('Sidecar equivalence failed at '+path)


def _catalog(inventory):
    if set(inventory)!=set(SPLITS):raise ValueError('Invalid sidecar split inventory')
    result={};seen=set()
    for split,entries in inventory.items():
        result[split]=[]
        for e in entries:
            if (set(e)!={'episode_id','samples','transition_indices_fingerprint'} or not isinstance(e['episode_id'],str)
                    or not e['episode_id'] or e['episode_id'] in seen or type(e['samples']) is not int or e['samples']<1
                    or not isinstance(e['transition_indices_fingerprint'],str)):
                raise ValueError('Invalid/duplicate sidecar episode inventory')
            seen.add(e['episode_id']);result[split].append(dict(e))
    return result


def _identity(identity):
    if set(identity)!=set(IDENTITIES) or any(type(v) is not str or not v for v in identity.values()):
        raise ValueError('Independent complete sidecar identities required')
    return dict(identity)


def _manifest_fingerprint(doc):return _hash({k:v for k,v in doc.items() if k!='manifest_fingerprint'})


def _read_manifest(root,expected_identity,*,allow_incomplete=False):
    try:doc=json.loads((root/'manifest.json').read_text())
    except (OSError,ValueError) as exc:raise ValueError('Missing/corrupt sidecar manifest') from exc
    if doc.get('schema_version')!=VERSION or (not allow_incomplete and doc.get('complete') is not True):
        raise ValueError('Unsupported/incomplete sidecar')
    if _identity(doc['identity'])!=_identity(expected_identity):raise ValueError('Stale/incompatible sidecar identity; explicit rebuild required')
    if doc.get('manifest_fingerprint')!=_manifest_fingerprint(doc):raise ValueError('Sidecar manifest checksum mismatch')
    if _catalog(doc['inventory'])!=doc['inventory']:raise ValueError('Invalid inventory')
    if doc['membership_fingerprint']!=_hash(doc['inventory']):raise ValueError('Invalid membership fingerprint')
    return doc


def _publish_manifest(root,doc):
    doc=dict(doc);doc['manifest_fingerprint']=_manifest_fingerprint(doc)
    temp=root/'manifest.json.tmp'
    with temp.open('w') as f:f.write(_json(doc)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,root/'manifest.json');return doc


def _layout(inventory,split):
    ends=[0];episodes=[]
    for e in inventory[split]:ends.append(ends[-1]+e['samples']);episodes.append(e['episode_id'])
    return episodes,ends


def _locate(inventory,split,index):
    if split not in SPLITS or type(index) is not int:raise ValueError('Invalid split/index')
    episodes,ends=_layout(inventory,split)
    if not 0<=index<ends[-1]:raise IndexError(index)
    i=bisect_right(ends,index)-1;return episodes[i],index-ends[i]


def _parts(sample):
    return ({'inputs':{k:v for k,v in sample['inputs'].items() if k in _STATIC_INPUTS},
             'routing':{k:v for k,v in sample['routing'].items() if k not in _DYNAMIC_ROUTING}},
            {'inputs':{k:v for k,v in sample['inputs'].items() if k not in _STATIC_INPUTS},
             'supervision':sample['supervision'],
             'routing':{k:v for k,v in sample['routing'].items() if k in _DYNAMIC_ROUTING}})


def _owned(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,dict):return {k:_owned(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_owned(v) for v in value)
    if isinstance(value,list):return [_owned(v) for v in value]
    # Frozen G2 typed routing objects and their immutable members may be shared.
    return value


def _join(template,dynamic):
    return {'inputs':{**_owned(template['inputs']),**dynamic['inputs']},
        'supervision':dynamic['supervision'],'routing':{**template['routing'],**dynamic['routing']}}


class SidecarWriter:
    """Single writer; committed records survive interruption. No implicit resume."""
    def __init__(self,root,identity,inventory,*,resume=False,commit_interval=64):
        if type(commit_interval) is not int or commit_interval<1:raise ValueError('Invalid commit interval')
        self.commit_interval=commit_interval;self.pending=0
        self.root=Path(root);self.stage=self.root.with_name(self.root.name+'.building')
        self.db=None;self.publication_ready=None
        self.raw_dynamic_bytes=0;self.raw_template_bytes=0;self.new_templates=0
        if self.root.exists() and not (resume and self.root.is_dir() and not self.root.is_symlink()
                and not any(self.root.iterdir()) and self.stage.is_dir()):raise ValueError('Sidecar output collision')
        self.identity=_identity(identity);self.inventory=_catalog(inventory)
        semantic={'schema_version':VERSION,'identity':self.identity,'inventory':self.inventory}
        self.base={**semantic,'complete':False,'membership_fingerprint':_hash(self.inventory),'semantic_fingerprint':_hash(semantic)}
        if resume:
            doc=_read_manifest(self.stage,self.identity,allow_incomplete=True)
            if doc['inventory']!=self.inventory or type(doc['complete']) is not bool:raise ValueError('Incompatible sidecar resume')
            if not (self.stage/'data.sqlite3').is_file():raise ValueError('Missing resume storage')
            if doc['complete'] is True:
                self.publication_ready=doc
                return
            if self.root.exists():raise ValueError('Incomplete sidecar cannot own a publication reservation')
        else:
            self.stage.mkdir(parents=True,exist_ok=False);_publish_manifest(self.stage,self.base)
        self.db=sqlite3.connect(self.stage/'data.sqlite3');self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA cache_size=-8192')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        if not resume:
            self.db.executescript('''
                CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE templates(id TEXT PRIMARY KEY,payload BLOB NOT NULL);
                CREATE TABLE samples(split TEXT NOT NULL,idx INTEGER NOT NULL,episode TEXT NOT NULL,
                    transition_idx INTEGER NOT NULL,local_idx INTEGER NOT NULL,template TEXT NOT NULL,
                    checksum TEXT NOT NULL,payload BLOB NOT NULL,structural INTEGER NOT NULL,
                    PRIMARY KEY(split,idx),UNIQUE(episode,transition_idx),UNIQUE(episode,local_idx),
                    FOREIGN KEY(template) REFERENCES templates(id));
                CREATE INDEX sample_identity ON samples(episode,transition_idx);
            ''')
            self.db.execute('INSERT INTO meta VALUES(?,?)',('semantic_fingerprint',self.base['semantic_fingerprint']));self.db.commit()
        elif self.db.execute('SELECT value FROM meta WHERE key=?',('semantic_fingerprint',)).fetchone()!=(self.base['semantic_fingerprint'],):
            self.close();raise ValueError('Resume storage identity mismatch')
        self.raw_dynamic_bytes=0;self.raw_template_bytes=0;self.new_templates=0

    def has(self,split,index):
        return self.db.execute('SELECT 1 FROM samples WHERE split=? AND idx=?',(split,index)).fetchone() is not None

    def add(self,sample,split,index):
        eid,local=_locate(self.inventory,split,index);audit=sample['routing']['audit']
        for route,identity in [('layout_fingerprint','layout_fingerprint'),('normalizer_fingerprint','f2_fingerprint')]:
            if sample['routing'][route]!=self.identity[identity]:raise ValueError('Sample upstream fingerprint/identity mismatch')
        if audit.episode_id!=eid or type(audit.transition_index) is not int or audit.transition_index<0:
            raise ValueError(f'Wrong logical sample identity at {split}/{index}')
        template,dynamic=_parts(sample);tb=encode(template);tid=_sha(tb)
        previous=self.db.execute('SELECT payload FROM templates WHERE id=?',(tid,)).fetchone()
        if previous is not None and previous[0]!=tb:raise ValueError('Template identity collision')
        if previous is None:
            self.db.execute('INSERT INTO templates VALUES(?,?)',(tid,tb));self.new_templates+=1
            assert_exact(decode(tb),template);self.raw_template_bytes+=len(zlib.decompress(tb))
        blob=encode(dynamic);assert_exact(decode(blob),dynamic)
        self.raw_dynamic_bytes+=len(zlib.decompress(blob))
        structural=bool(sample['supervision']['structural']['supervised'].any())
        try:self.db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?)',
            (split,index,eid,audit.transition_index,local,tid,_sha(blob),blob,int(structural)))
        except sqlite3.IntegrityError as exc:raise ValueError(f'Duplicate/invalid sample {eid}/{audit.transition_index}') from exc
        self.pending+=1
        if self.pending==self.commit_interval:self.db.commit();self.pending=0

    def finalize(self):
        if self.publication_ready is not None:
            verify_sidecar(self.stage,expected_identity=self.identity)
            return self._finish_publication(self.publication_ready)
        self.db.commit();counts={};episodes=0
        for split,entries in self.inventory.items():
            counts[split]={'episodes':len(entries),'samples':sum(e['samples'] for e in entries)};episodes+=len(entries)
            for e in entries:
                rows=self.db.execute('SELECT local_idx,transition_idx FROM samples WHERE episode=? ORDER BY local_idx',(e['episode_id'],)).fetchall()
                if ([r[0] for r in rows]!=list(range(e['samples'])) or _hash([r[1] for r in rows])!=e['transition_indices_fingerprint']):
                    raise ValueError('Incomplete/incorrect sidecar episode '+e['episode_id'])
        total=sum(c['samples'] for c in counts.values())
        if self.db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]!=total:raise ValueError('Unexpected sidecar records')
        if self.db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('Corrupt SQLite sidecar')
        templates=self.db.execute('SELECT COUNT(*) FROM templates').fetchone()[0]
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.close();size=(self.stage/'data.sqlite3').stat().st_size
        with (self.stage/'data.sqlite3').open('rb') as f:os.fsync(f.fileno())
        doc=_publish_manifest(self.stage,{**self.base,'complete':True,'episode_count':episodes,'sample_count':total,
            'split_counts':counts,'template_count':templates,'storage':{'format':'sqlite3+zlib1+pickle5-packed-tensors',
                'file':'data.sqlite3','bytes':size},'verification':'each dynamic record and each distinct template roundtripped exactly'})
        self.publication_ready=doc
        return self._finish_publication(doc)

    def _finish_publication(self,doc):
        # Explicit resume may adopt only the empty reservation paired with this
        # certified complete staging artifact. Never overwrite a nonempty path.
        if self.root.exists():
            if self.root.is_symlink() or not self.root.is_dir() or any(self.root.iterdir()):raise ValueError('Sidecar publication collision')
        else:self.root.mkdir()
        descriptor=os.open(self.stage,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        os.rename(self.stage,self.root)
        descriptor=os.open(self.root.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        self.publication_ready=None
        return doc

    def import_episode(self,root,split,episode_id):
        """Atomically merge an owned worker artifact; no tensor recomputation."""
        with SidecarDataset(root,split,expected_identity=self.identity) as child:
            expected=next(e for e in self.inventory[split] if e['episode_id']==episode_id)
            if child.manifest['inventory']!={s:([expected] if s==split else []) for s in SPLITS}:
                raise ValueError('Worker membership mismatch')
        offset=0
        for e in self.inventory[split]:
            if e['episode_id']==episode_id:break
            offset+=e['samples']
        self.db.commit();self.pending=0
        self.db.execute('ATTACH DATABASE ? AS worker',(str(Path(root)/'data.sqlite3'),))
        try:
            with self.db:
                if self.db.execute('SELECT COUNT(*) FROM templates m JOIN worker.templates w ON m.id=w.id WHERE m.payload!=w.payload').fetchone()[0]:
                    raise ValueError('Worker template collision')
                self.db.execute('INSERT OR IGNORE INTO templates SELECT * FROM worker.templates')
                self.db.execute('INSERT INTO samples SELECT split,idx+?,episode,transition_idx,local_idx,template,checksum,payload,structural FROM worker.samples',(offset,))
        except sqlite3.Error as exc:raise ValueError('Invalid/duplicate worker episode '+episode_id) from exc
        finally:self.db.execute('DETACH DATABASE worker')

    def close(self):
        if getattr(self,'db',None) is not None:self.db.commit();self.db.close();self.db=None


class SidecarDataset:
    """Direct indexed lookup, bounded private template LRU; no G1/F2/G2 calls."""
    def __init__(self,root,split,*,expected_identity,template_cache_size=16):
        self.root=Path(root)
        if split not in SPLITS or type(template_cache_size) is not int or not 0<=template_cache_size<=64:
            raise ValueError('Invalid sidecar split/cache size')
        self.manifest=_read_manifest(self.root,expected_identity);self.split=split;self.limit=template_cache_size
        storage=self.manifest['storage']
        if storage['file']!='data.sqlite3':raise ValueError('Invalid sidecar storage path')
        path=self.root/storage['file']
        if not path.is_file() or path.stat().st_size!=storage['bytes']:raise ValueError('Missing/truncated sidecar storage')
        self.db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);self.db.execute('PRAGMA cache_size=-8192')
        try:
            tables={r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'meta','samples','templates'}<=tables:raise ValueError('Missing sidecar storage table')
            if self.db.execute('SELECT value FROM meta WHERE key=?',('semantic_fingerprint',)).fetchone()!=(self.manifest['semantic_fingerprint'],):
                raise ValueError('Incompatible sidecar storage identity')
        except Exception as exc:self.db.close();raise ValueError('Invalid sidecar storage') from exc
        self.templates=OrderedDict();self.episode_ids,self.ends=_layout(self.manifest['inventory'],split)

    def __len__(self):return self.ends[-1]
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        if getattr(self,'db',None) is not None:self.db.close();self.db=None
        self.templates.clear()

    def __getitem__(self,index):
        if type(index) is not int:raise TypeError('Integer sidecar index required')
        if index<0:index+=len(self)
        eid,local=_locate(self.manifest['inventory'],self.split,index)
        try:
            row=self.db.execute('SELECT episode,transition_idx,local_idx,template,checksum,payload FROM samples WHERE split=? AND idx=?',(self.split,index)).fetchone()
            if row is None:raise ValueError('Missing sample record')
            episode,transition,ordinal,tid,checksum,blob=row
            if (episode,ordinal)!=(eid,local):raise ValueError('Invalid split/index mapping')
            if _sha(blob)!=checksum:raise ValueError('Sample checksum mismatch')
            if tid in self.templates:
                template=self.templates[tid];self.templates.move_to_end(tid)
            else:
                record=self.db.execute('SELECT payload FROM templates WHERE id=?',(tid,)).fetchone()
                if record is None or _sha(record[0])!=tid:raise ValueError('Missing/corrupt template')
                template=decode(record[0])
                if self.limit:
                    self.templates[tid]=template
                    if len(self.templates)>self.limit:self.templates.popitem(last=False)
            result=_join(template,decode(blob));audit=result['routing']['audit']
            if (audit.episode_id,audit.transition_index)!=(episode,transition):raise ValueError('Record routing identity mismatch')
            for route,identity in [('layout_fingerprint','layout_fingerprint'),('normalizer_fingerprint','f2_fingerprint')]:
                if result['routing'][route]!=self.manifest['identity'][identity]:raise ValueError('Record upstream identity mismatch')
            return result
        except Exception as exc:raise ValueError(f'Cannot read sidecar sample {self.split}/{index} ({eid}, local {local}): {exc}') from exc

    def get(self,episode_id,transition_index):
        row=self.db.execute('SELECT idx FROM samples WHERE split=? AND episode=? AND transition_idx=?',
            (self.split,episode_id,transition_index)).fetchone()
        if row is None:raise ValueError(f'Missing sidecar sample {episode_id}/{transition_index} in {self.split}')
        return self[row[0]]

    def sample_keys(self):
        count=0
        for index,eid,transition,local in self.db.execute('SELECT idx,episode,transition_idx,local_idx FROM samples WHERE split=? ORDER BY idx',(self.split,)):
            if index!=count or _locate(self.manifest['inventory'],self.split,index)!=(eid,local):
                raise ValueError('Invalid sidecar split/index coverage')
            count+=1;yield eid,transition
        if count!=len(self):raise ValueError('Incomplete sidecar split coverage')

    def train_index(self):
        from .sampling import TrainRowIndex
        if self.split!='train':raise ValueError('TRAIN index required')
        for _ in self.sample_keys():pass  # Reject missing tails before sampling.
        episodes={e:[] for e in self.episode_ids};structural={}
        for i,eid,rare in self.db.execute('SELECT idx,episode,structural FROM samples WHERE split=? ORDER BY idx',('train',)):
            if _locate(self.manifest['inventory'],'train',i)[0]!=eid:raise ValueError('Invalid TRAIN mapping')
            episodes[eid].append(i)
            if rare:structural.setdefault(eid,[]).append(i)
        ids=self.manifest['identity']
        return TrainRowIndex(episodes,structural,{'split_name':'train','g1':ids['g1_fingerprint'],
            'split':ids['split_fingerprint'],'f2':ids['f2_fingerprint'],'tensor_layout':ids['layout_fingerprint']})


def verify_sidecar(root,*,expected_identity):
    """Explicit streaming deep check of every split/record/template, no model."""
    from collections import Counter
    seen=set();counts={};operations=Counter();structural_samples=0;episodes=set()
    for split in SPLITS:
        with SidecarDataset(root,split,expected_identity=expected_identity) as data:
            if data.db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('Corrupt sidecar database')
            keys=list(data.sample_keys());counts[split]=len(keys)
            for i,key in enumerate(keys):
                if key in seen:raise ValueError('Duplicate logical sidecar identity')
                seen.add(key);episodes.add(key[0]);sample=data[i]
                targets=sample['routing']['structural_targets']
                valid=[t for t in targets if t.supervised]
                operations.update(t.operation_id for t in valid);structural_samples+=bool(valid)
            for e in data.manifest['inventory'][split]:
                transitions=[t for eid,t in keys if eid==e['episode_id']]
                if len(transitions)!=e['samples'] or _hash(transitions)!=e['transition_indices_fingerprint']:
                    raise ValueError('Incorrect Canonical transition coverage '+e['episode_id'])
    return {'samples':len(seen),'episodes':len(episodes),'duplicates':0,'missing':0,'routing_failures':0,
        'splits':counts,'structural_samples':structural_samples,'operations':dict(sorted(operations.items()))}
