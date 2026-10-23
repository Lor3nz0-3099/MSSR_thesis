"""Independent physical action membership and value codecs over frozen H1 ABI.

All resource candidates precede all structural candidates across a batch. One
shared scorer predicts membership for both typed embedding families; payload,
argument and scope outputs retain their existing catalog value codecs. Only
current numeric inputs and H1 latents cross this neural boundary.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .action_heads import ActionHeads, _mlp
from .model import BCGNN, _owners, _indices, _pointer


def _validate_config(config):
    if (set(config) != {'schema_version', 'hidden_dim', 'activation', 'dropout'}
            or config['schema_version'] != 'mssr.bc_v1.action_set_head.v1'
            or type(config['hidden_dim']) is not int or config['hidden_dim'] != 128
            or config['activation'] != 'silu'
            or type(config['dropout']) is not float or config['dropout'] != .1):
        raise ValueError('Unsupported ActionSet head configuration')


def load_action_set_head_config(path):
    config = json.loads(Path(path).read_text())
    _validate_config(config)
    return config


class ActionSetHead(ActionHeads):
    """Catalog conditioning and value codecs with one shared presence head."""

    def __init__(self, spec, latent_config, config):
        _validate_config(config)
        # Reuse the established value codec/catalog construction, retaining no
        # trainable mandatory decision or absence scorer in the resulting head.
        legacy = {**config, 'schema_version': 'mssr.bc_v1.action_heads.v1'}
        super().__init__(spec, latent_config, legacy)
        self.config = dict(config)
        del self.mode_scorer
        del self.structural_scorer
        del self.no_structural_scorer
        del self.outcome_projection
        del self.resource_contract
        self.presence_scorer = _mlp(config['hidden_dim'], 1, config)
        for layer in self.presence_scorer.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _resource_candidate_conditions(self, latents, inputs):
        rows = inputs['resource_candidates']
        r = len(inputs['resource_x'])
        q = len(rows['resource'])
        _pointer(rows['group_ptr'], q, 'resource candidate groups')
        if len(rows['group_ptr']) != r + 1:
            raise ValueError('Invalid resource candidate groups')
        _indices(rows['resource'], r, 'resource candidate')
        _indices(rows['mode'], len(self.modes), 'resource mode')
        if rows['mode'].shape != (q,):
            raise ValueError('Resource mode shape mismatch')
        if not torch.equal(_owners(rows['group_ptr']), rows['resource']):
            raise ValueError('Resource candidate/group mismatch')
        sizes = self._catalog_mode_width[rows['mode']]
        expected = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)))
        if not torch.equal(rows['payload_ptr'], expected):
            raise ValueError('Catalog payload arity mismatch')
        conditioned = self._resource_conditions(latents, inputs)[rows['resource']]
        if q:
            conditioned = conditioned + self.mode_projection(F.one_hot(rows['mode'], len(self.modes)).float())
        return conditioned

    def _resource_parameters(self, conditions, inputs):
        """Predict values for every candidate without selecting a mode first."""
        rows = inputs['resource_candidates']
        sizes = self._catalog_mode_width[rows['mode']]
        numeric = self._catalog_mode_numeric[rows['mode']].bool()
        payload_rows = torch.nonzero(numeric, as_tuple=False).flatten()
        payload_ptr = torch.cat((sizes.new_zeros(1), sizes[payload_rows].cumsum(0)))
        owner = _owners(payload_ptr)
        query = payload_rows[owner]
        components = (self._catalog_mode_component_start[rows['mode'][query]]
            + torch.arange(len(query), device=query.device) - payload_ptr[owner])
        payload = conditions.new_empty(0)
        if len(query):
            role = self.component_projection(F.one_hot(components, len(self.payload_components)).float())
            payload = self.payload_decoder(conditions[query] + role).flatten()
        category_rows = torch.nonzero(~numeric, as_tuple=False).flatten()
        widths = self._catalog_mode_categories[rows['mode'][category_rows]]
        category_ptr = torch.cat((widths.new_zeros(1), widths.cumsum(0)))
        owner = _owners(category_ptr)
        query = category_rows[owner]
        options = (self._catalog_mode_category_start[rows['mode'][query]]
            + torch.arange(len(query), device=query.device) - category_ptr[owner])
        category_logits = conditions.new_empty(0)
        if len(query):
            role = self.resource_category_projection(F.one_hot(options, len(self.resource_categories)).float())
            category_logits = self.resource_category_scorer(conditions[query] + role).flatten()
        return {'payload': payload, 'payload_candidate': payload_rows,
            'payload_ptr': payload_ptr, 'categorical': {'candidate': category_rows,
                'ptr': category_ptr, 'logits': category_logits}}

    def forward_resource(self, latents, inputs):
        """Resource value codecs only; no mandatory resource decision."""
        self._validate_input_boundary(inputs)
        return self._resource_parameters(self._resource_candidate_conditions(latents, inputs), inputs)

    def forward_structural(self, latents, inputs):
        """Structural value codecs only; membership lives in shared forward."""
        self._validate_input_boundary(inputs)
        conditions = self._structural_conditions(latents, inputs)
        return {'arguments': self._arguments(latents, inputs, conditions),
            'scope': self._scope(latents, inputs, conditions)}

    def forward(self, latents, inputs):
        self._validate_input_boundary(inputs)
        resource = self._resource_candidate_conditions(latents, inputs)
        structural = self._structural_conditions(latents, inputs)
        conditions = torch.cat((resource, structural), dim=0)
        module_graph = _owners(inputs['graph_ptr'])
        resource_graph = module_graph[inputs['resource_to_module'][inputs['resource_candidates']['resource']]]
        structural_graph = module_graph[inputs['structural']['initiator']]
        return {'presence_logits': self.presence_scorer(conditions).flatten(),
            'candidate_graph': torch.cat((resource_graph, structural_graph)),
            'parameters': {'resource': self._resource_parameters(resource, inputs),
                'structural': {'arguments': self._arguments(latents, inputs, structural),
                    'scope': self._scope(latents, inputs, structural)}}}


class ActionSetPolicy(nn.Module):
    """Compose unchanged standalone H1 with independent set membership H2."""

    def __init__(self, spec, schema, h1_config, head_config):
        super().__init__()
        self.backbone = BCGNN(spec, schema, h1_config)
        self.head = ActionSetHead(spec, h1_config, head_config)

    def forward(self, inputs):
        return self.head(self.backbone.encode(inputs), inputs)
