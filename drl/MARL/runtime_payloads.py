"""Lazy distribution views over the SAME live H2 output; no value caching.

Only integer routing descriptors are retained across calls, bounded by the
certified static layout token. A new prediction owns fresh lazy mappings.
"""
from collections import OrderedDict
from collections.abc import Mapping

from drl.MARL.policy import (
    ResourcePayloadView, StructuralPayloadView, StructuralNumericArgumentView,
    StructuralBooleanArgumentView, StructuralCategoricalArgumentView,
    StructuralReferenceArgumentView, StructuralReferenceSetArgumentView, StructuralScopeView,
)


def _host(tensor):
    return tuple(tensor.detach().cpu().tolist())


class LazyPayloadMapping(Mapping):
    def __init__(self, indices, build):
        self._indices = tuple(indices)
        self._valid = frozenset(self._indices)
        self._build = build
        self._memo = {}

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if index not in self._valid:
            raise KeyError(index)
        if index not in self._memo:
            self._memo[index] = self._build(index)
        return self._memo[index]

    @property
    def materialized_count(self):
        return len(self._memo)

    def with_availability(self, availability):
        from drl.MARL.c1_feasibility import restrict_structural_payload
        scopes = dict(availability.scope_blocked)
        refs = {(i,a): blocked for i,a,blocked in availability.reference_set_blocked}
        return LazyPayloadMapping(self._indices,
            lambda index: restrict_structural_payload(self[index], index, scopes, refs))


class RuntimePayloadCache:
    def __init__(self, max_layouts=8):
        self.max_layouts = max_layouts
        self._layouts = OrderedDict()
        self.builds = self.hits = 0

    def _metadata(self, parameters, graph):
        resource = parameters['resource']
        numeric_ptr = _host(resource['payload_ptr'])
        numeric = {index: numeric_ptr[row:row+2] for row,index in enumerate(_host(resource['payload_candidate']))}
        categories = resource['categorical'];ptr = _host(categories['ptr'])
        categorical = {index: ptr[row:row+2] for row,index in enumerate(_host(categories['candidate']))}
        structural = parameters['structural'];bindings = _host(graph['structural']['argument_binding'])
        families = {}
        for family, head in structural['arguments'].items():
            name = 'category' if family == 'categorical' else family
            grouped = {}
            arguments = _host(head['argument'])
            ptr = _host(head['ptr']) if name != 'boolean' else ()
            required = _host(head['required_ptr']) if name == 'reference_set' else ()
            for row, argument in enumerate(arguments):
                descriptor = (argument,row) if name == 'boolean' else (argument,row,*ptr[row:row+2])
                if required:
                    descriptor = (*descriptor,*required[row:row+2])
                grouped.setdefault(bindings[argument], []).append(descriptor)
            families[name] = {index:tuple(rows) for index,rows in grouped.items()}
        scope = structural['scope'];ptr,required = _host(scope['ptr']),_host(scope['required_ptr'])
        scopes = {binding:(*ptr[row:row+2],*required[row:row+2])
                  for row,binding in enumerate(_host(scope['binding']))}
        self.builds += 1
        return numeric,categorical,families,scopes

    def views(self, *, parameters, graph, routing):
        token = routing.get('runtime_candidate_layout_token')
        meta = self._layouts.pop(token, None) if token is not None else None
        if meta is None:
            meta = self._metadata(parameters, graph)
        else:
            self.hits += 1
        if token is not None:
            self._layouts[token] = meta
            while len(self._layouts) > self.max_layouts:
                self._layouts.popitem(last=False)
        numeric,categorical,families,scopes = meta
        resource = parameters['resource'];structural = parameters['structural'];arguments = structural['arguments']
        resource_count = len(routing['resource_candidates'])

        def resource_view(index):
            if index in categorical:
                a,b = categorical[index]
                return ResourcePayloadView(categorical_logits=resource['categorical']['logits'][a:b])
            a,b = numeric[index]
            return ResourcePayloadView(numeric=resource['payload'][a:b])

        def structural_view(index):
            binding = index-resource_count
            values = {}
            for family,grouped in families.items():
                name = 'category' if family == 'category' else family
                head = arguments.get(name,arguments.get('categorical'))
                views = []
                for descriptor in grouped.get(binding, ()):
                    argument,row,*parts = descriptor
                    if family == 'boolean':
                        view = StructuralBooleanArgumentView(argument,head['logits'][row])
                    elif family == 'numeric':
                        a,b = parts;view = StructuralNumericArgumentView(argument,head['value'][a:b])
                    elif family == 'category':
                        a,b = parts;view = StructuralCategoricalArgumentView(argument,head['logits'][a:b])
                    elif family == 'reference':
                        a,b = parts;view = StructuralReferenceArgumentView(argument,head['logits'][a:b],head['domain_index'][a:b])
                    elif family == 'reference_set':
                        a,b,c,d = parts
                        view = StructuralReferenceSetArgumentView(argument,head['logits'][a:b],head['domain_index'][a:b],head['required_domain_index'][c:d])
                    else:
                        raise ValueError(f'Unknown structural payload family: {family}')
                    views.append(view)
                values['categorical' if family == 'category' else family] = tuple(views)
            if binding in scopes:
                a,b,c,d = scopes[binding];head = structural['scope']
                values['scope'] = StructuralScopeView(head['logits'][a:b],head['domain_index'][a:b],head['required_domain_index'][c:d])
            return StructuralPayloadView(**values)

        resource_indices = tuple(dict.fromkeys((*numeric,*categorical)))
        return (LazyPayloadMapping(resource_indices,resource_view),
                LazyPayloadMapping(range(resource_count, resource_count+len(routing['structural_bindings'])),structural_view))
