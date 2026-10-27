"""Optional fold-specific tensorization over a shared, bounded physical G1 cache.

No fitting, G2 rebuilding or inverse normalization. The injected reader must
return authenticated immutable G1 samples. Frozen G2 supplies the reviewed v3
authority/masks; all nonnumeric routing and supervision must still match.
"""
import torch

from .training_data import _BudgetLRU, _retained_size
from .tensorization import tensorize_policy_sample
from .normalization import normalizer_fingerprint
from .resource_decisions import add_resource_decisions
from .structural_decisions import add_no_structural_supervision


class RawSampleCache:
    def __init__(self, g1_fingerprint, *, budget_bytes, reader):
        if not isinstance(g1_fingerprint,str) or not g1_fingerprint:
            raise ValueError('Authenticated G1 identity required')
        if type(budget_bytes) is not int or budget_bytes<0 or not callable(reader):
            raise ValueError('Bounded raw cache and physical reader required')
        self.g1_fingerprint=g1_fingerprint
        self._cache=_BudgetLRU(budget_bytes);self._reader=reader

    def get(self, audit):
        key=(self.g1_fingerprint,audit.episode_id,audit.transition_index)
        sample=self._cache.get(key)
        if sample is None:
            sample=self._reader(audit.episode_id,audit.transition_index)
            if sample.audit!=audit:raise ValueError('Physical source sample audit mismatch')
            self._cache.put(key,sample,_retained_size(sample))
        if sample.audit!=audit:raise ValueError('Cached physical sample audit mismatch')
        return sample

    def stats(self):
        return {'entries':len(self._cache.entries),'retained_bytes':self._cache.bytes,
                'hits':self._cache.hits,'budget_bytes':self._cache.budget}

    def close(self):self._cache.clear()


def _same_nonnumeric(first, second, *, numeric_keys=()):
    if first.keys()!=second.keys():raise ValueError('Fold tensor schema changed')
    for key,value in first.items():
        other=second[key]
        if key in numeric_keys:
            if value.dtype!=other.dtype or value.shape!=other.shape:
                raise ValueError('Fold numeric routing changed')
        elif not torch.equal(value,other):
            raise ValueError('Fold candidate/supervision routing changed: '+key)


class FoldNormalizedView:
    """Non-owning DEV view; source/cache lifecycle remains with the caller.

    CV split membership and train-only fit provenance must be validated by the
    future CV coordinator before constructing views. This class performs no CV.
    """
    def __init__(self, source, raw_cache, *, contract, spec, feature_schema,
                 normalization_policy, normalizers, tensor_layout):
        if source.split not in ('train','validation'):
            raise ValueError('Fold views forbid TEST')
        identity=source.manifest['identity']
        if identity['g1_fingerprint']!=raw_cache.g1_fingerprint:
            raise ValueError('Raw cache G1 identity mismatch')
        self.source,self.raw_cache,self.split=source,raw_cache,source.split
        self.contract=contract
        self.kwargs=dict(spec=spec,feature_schema=feature_schema,normalization_policy=normalization_policy,
                         normalizers=normalizers,tensor_layout=tensor_layout)
        self.cache_identity={'source_manifest':source.manifest['manifest_fingerprint'],
            'g1':raw_cache.g1_fingerprint,'f2':normalizer_fingerprint(normalizers),
            'layout':tensor_layout['fingerprint'],'resource_contract':contract.fingerprint}

    def __len__(self):return len(self.source)

    def __getitem__(self,index):
        frozen=self.source[index]
        raw=self.raw_cache.get(frozen['routing']['audit'])
        row=add_no_structural_supervision(tensorize_policy_sample(raw,**self.kwargs))
        y=frozen['supervision']['resource']
        # Authority is never inferred from a missing label: use ONLY the frozen
        # explicitly supervised INACTIVE decisions. Positive/masked physical
        # rows come directly from the immutable raw supervision.
        x=frozen['inputs']['resource_candidates'];authority=set()
        for group,candidate,valid in zip(y['decision_group'].tolist(),y['decision_candidate'].tolist(),y['decision_supervised'].tolist()):
            if valid and candidate>=0 and int(x['outcome_kind'][candidate])==0:authority.add(group)
        row=add_resource_decisions(row,self.contract,authoritative_resources=authority)
        for family in ('resource_candidates','structural'):
            _same_nonnumeric(row['inputs'][family],frozen['inputs'][family])
        for family,numeric in [('resource',('payload',)),('structural',('argument_value',))]:
            _same_nonnumeric(row['supervision'][family],frozen['supervision'][family],numeric_keys=numeric)
        for key in ('module_refs','connector_refs','resource_refs','coordinate_refs','connection_refs','relation_refs'):
            if row['routing'][key]!=frozen['routing'][key]:raise ValueError('Fold entity routing changed')
        return row

    get_for_collate=__getitem__
