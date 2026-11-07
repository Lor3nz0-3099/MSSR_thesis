"""Closed-ABI H2 scatter/gather tests; independent of Isaac and network config."""

from __future__ import annotations

import torch

from drl.MARL.h2_batching import split_action_set_h2_batch


def _make_head(indices, widths, *, values_key="logits", domain=False, required=False):
    """One unbatched family with deterministic, differentiable values."""
    indices = torch.tensor(indices, dtype=torch.int64)
    ptr = torch.tensor([0, *torch.tensor(widths).cumsum(0).tolist()], dtype=torch.int64)
    values = torch.randn(int(ptr[-1]), requires_grad=True)
    output = {"argument": indices, "ptr": ptr, values_key: values}
    if domain:
        output["domain_index"] = torch.arange(int(ptr[-1]), dtype=torch.int64)
    if required:
        required_widths = [1 if w else 0 for w in widths]
        req_ptr = torch.tensor([0, *torch.tensor(required_widths).cumsum(0).tolist()],dtype=torch.int64)
        output["required_ptr"] = req_ptr
        output["required_domain_index"] = torch.arange(int(req_ptr[-1]), dtype=torch.int64)
    return output


def _make_local(q, s, a, domain_count, scope_count):
    # All index domains are local to their own input graph.
    resource_numeric = [i for i in range(q) if i % 2 == 0]
    resource_cat = [i for i in range(q) if i % 2 == 1]
    arg_ref = [i for i in range(a) if i % 5 == 3]
    arg_ref_set = [i for i in range(a) if i % 5 == 4]
    numeric = _make_head(resource_numeric,[1] * len(resource_numeric),values_key="value")
    categorical = _make_head(resource_cat,[2] * len(resource_cat))
    nrows = list(range(a))
    family_indices = {
        "numeric": [i for i in nrows if i % 5 == 0],
        "category": [i for i in nrows if i % 5 == 1],
        "boolean": [i for i in nrows if i % 5 == 2],
        "reference": arg_ref,
        "reference_set": arg_ref_set,
    }
    arguments = {}
    for family, ids in family_indices.items():
        if family == "boolean":
            arguments[family] = {
                "argument": torch.tensor(ids, dtype=torch.int64),
                "logits": torch.randn(len(ids), requires_grad=True),
            }
            continue
        d = family in ("reference","reference_set")
        r = family == "reference_set"
        entry = _make_head(ids, [1] * len(ids) if d else [2] * len(ids),
                           values_key="value" if family == "numeric" else "logits",
                           domain=d,required=r)
        # The reference indices come from the flat *input* argument domain.
        if d:
            entry["domain_index"] = torch.arange(len(entry["domain_index"]),dtype=torch.int64) % max(domain_count, 1)
            if r:
                entry["required_domain_index"] = torch.arange(len(entry["required_domain_index"]),dtype=torch.int64) % max(domain_count, 1)
        arguments[family] = entry
    scope = _make_head([i for i in range(s) if i % 2 == 0], [1] * ((s+1)//2), domain=True,required=True)
    scope["binding"] = scope.pop("argument")
    scope["domain_index"] = torch.arange(len(scope["domain_index"]),dtype=torch.int64) % max(scope_count, 1)
    scope["required_domain_index"] = torch.arange(len(scope["required_domain_index"]),dtype=torch.int64) % max(scope_count, 1)
    return {
        "presence_logits": torch.randn(q+s,requires_grad=True),
        "candidate_graph": torch.zeros(q+s,dtype=torch.int64),
        "parameters": {
            "resource": {
                "payload": numeric["value"],
                "payload_candidate": numeric["argument"],
                "payload_ptr": numeric["ptr"],
                "categorical": {"candidate": categorical["argument"],"ptr":categorical["ptr"],"logits":categorical["logits"]},
            },
            "structural": {"arguments": arguments,"scope":scope},
        },
    }


def _packed_head(rows, *, index, index_offsets, values_key, domain_offset=None, required=False, ptr=True):
    output = {index: torch.cat([r[index]+off for r,off in zip(rows,index_offsets)])}
    if ptr:
        ptrs = [r["ptr"] for r in rows]
        count = 0
        allptr = [torch.zeros(1,dtype=torch.int64)]
        for p in ptrs:
            allptr.append(p[1:]+count)
            count += int(p[-1])
        output["ptr"] = torch.cat(allptr)
    output[values_key] = torch.cat([r[values_key] for r in rows])
    if domain_offset is not None:
        output["domain_index"] = torch.cat([r["domain_index"]+off for r,off in zip(rows,domain_offset)])
    if required:
        count=0
        rptrs = [torch.zeros(1,dtype=torch.int64)]
        for r in rows:
            rptrs.append(r["required_ptr"][1:]+count)
            count += int(r["required_ptr"][-1])
        output["required_ptr"] = torch.cat(rptrs)
        output["required_domain_index"] = torch.cat([r["required_domain_index"]+off for r,off in zip(rows,domain_offset)])
    return output


def _pack(local, qs, ss, aa, domains, scopes):
    def pref(xs):
        return [0,*torch.tensor(xs).cumsum(0).tolist()]
    ro, so, ao, do, sco = (pref(x)[:-1] for x in (qs,ss,aa,domains,scopes))
    resource = [x["parameters"]["resource"] for x in local]
    structural = [x["parameters"]["structural"] for x in local]
    numeric_rows = [dict(candidate=r["payload_candidate"],ptr=r["payload_ptr"],value=r["payload"]) for r in resource]
    nr = _packed_head(numeric_rows,index="candidate",index_offsets=ro,values_key="value")
    cats = _packed_head([r["categorical"] for r in resource],index="candidate",index_offsets=ro,values_key="logits")
    args = {name:_packed_head([s["arguments"][name] for s in structural], index="argument",index_offsets=ao,
                              values_key="value" if name=="numeric" else "logits",
                              domain_offset=do if name in ("reference","reference_set") else None,
                              required=name=="reference_set",ptr=name!="boolean")
            for name in ("numeric","category","boolean","reference","reference_set")}
    scope = _packed_head([s["scope"] for s in structural],index="binding",index_offsets=so,
                         values_key="logits",domain_offset=sco,required=True)
    presence = torch.cat([torch.cat([l["presence_logits"][:q] for l,q in zip(local,qs)]),
                          torch.cat([l["presence_logits"][q:] for l,q in zip(local,qs)])])
    candidate_graph = torch.cat([torch.repeat_interleave(torch.arange(len(local)),torch.tensor(qs)),
                                 torch.repeat_interleave(torch.arange(len(local)),torch.tensor(ss))])
    return {"presence_logits": presence,"candidate_graph":candidate_graph,
            "parameters":{"resource":{"payload":nr["value"],"payload_candidate":nr["candidate"],"payload_ptr":nr["ptr"],"categorical":cats},
                          "structural":{"arguments":args,"scope":scope}}}


def _tree_equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:_tree_equal(a[k],b[k])
    elif torch.is_tensor(a):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    else:
        assert a==b


def _values(obj):
    if isinstance(obj,dict):
        for v in obj.values():
            yield from _values(v)
    elif torch.is_tensor(obj) and obj.dtype.is_floating_point:
        yield obj


def test_h2_split_all_typed_families_and_gradients():
    qs,ss,aa,dd,sc = [3,0,4],[2,1,3],[5,0,7],[12,1,10],[3,2,8]
    # Require domains sufficiently large for locally indexed test fixtures.
    graphs = [
        {"resource_candidates":{"resource":torch.zeros(q,dtype=torch.int64)},
         "structural":{"initiator":torch.zeros(s,dtype=torch.int64),
                       "argument_binding":torch.zeros(a,dtype=torch.int64),
                       "argument_domain_index":torch.zeros(d,dtype=torch.int64),
                       "scope_members":torch.zeros(z,dtype=torch.int64)}}
        for q,s,a,d,z in zip(qs,ss,aa,dd,sc)
    ]
    local = [_make_local(*vals) for vals in zip(qs,ss,aa,dd,sc)]
    packed = _pack(local,qs,ss,aa,dd,sc)
    split = split_action_set_h2_batch(packed=packed,graphs=graphs)
    assert len(split)==len(local)
    for expected, actual in zip(local,split):
        _tree_equal(expected,actual)
    target = sum((v.square().sum() for item in local for v in _values(item)))
    actual = sum((v.square().sum() for item in split for v in _values(item)))
    torch.testing.assert_close(target,actual,atol=1e-5,rtol=1e-5)
    leaves=[v for item in local for v in _values(item)]
    ga=torch.autograd.grad(target,leaves,retain_graph=True)
    gb=torch.autograd.grad(actual,leaves)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
