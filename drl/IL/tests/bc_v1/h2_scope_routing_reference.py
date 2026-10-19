"""Test-only P1 scope/routing oracle; never imported by production."""
import torch
from torch.nn import functional as F
from drl.IL.bc_v1.action_heads import _ints


class PreP1bRouting:

    def _scope(self,latents,inputs,conditions):
            """Independent participant membership, never an argument-set alias.

            Fully determined scopes need no prediction. Required members for a
            variable scope are routing only; logits cover its optional members.
            domain_index indexes the existing flat structural scope_members table.
            """
            s=inputs['structural'];bindings=[];optional=[];required=[];ptr=[0];required_ptr=[0];scores=[]
            for binding in range(len(s['initiator'])):
                start,end=s['scope_ptr'][binding:binding+2].tolist();mask=s['scope_required'][start:end]
                if mask.all():continue
                domain=torch.arange(start,end,device=mask.device);chosen=domain[~mask]
                if self.scope_scorer is None:raise ValueError('Catalog has no variable participant scope')
                members=s['scope_members'][chosen]
                scores.append(self.scope_scorer(conditions[binding,None]+
                    self.scope_option_projection(latents['module_h'][members])).flatten())
                bindings.append(binding);optional.extend(chosen.tolist());required.extend(domain[mask].tolist())
                ptr.append(len(optional));required_ptr.append(len(required))
            return {'binding':_ints(bindings,s['initiator']),'logits':torch.cat(scores) if scores else conditions.new_empty(0),
                'ptr':_ints(ptr,s['initiator']),'domain_index':_ints(optional,s['initiator']),
                'required_ptr':_ints(required_ptr,s['initiator']),'required_domain_index':_ints(required,s['initiator'])}

    def _parameter_conditions(self,conditions,s,arguments):
            arguments=_ints(arguments,s['argument_binding'])
            params=[self.learnable_index[int(s['argument_parameter'][i])] for i in arguments.tolist()]
            result=conditions[s['argument_binding'][arguments]]
            if len(arguments):result=result+self.parameter_projection(F.one_hot(_ints(params,arguments),len(self.learnable_parameters)).float())
            return arguments,result

    def _arguments(self,latents,inputs,conditions):
            self._validate_arguments(inputs);s=inputs['structural']
            selected={key:[] for key in ('numeric','reference_set','reference','boolean','category')}
            for i,parameter_index in enumerate(s['argument_parameter'].tolist()):
                p=self.parameters_catalog[parameter_index]
                if not p.learnable:continue
                if p.kind in ('scalar','vector'):selected['numeric'].append(i)
                if p.kind in ('module_refs','connector_refs'):selected['reference_set'].append(i)
                if p.kind in ('module_ref','connector_ref') and int(s['argument_bound_ref'][i])==-1:selected['reference'].append(i)
                if p.kind in ('boolean','category'):selected[p.kind].append(i)
            arguments,base=self._parameter_conditions(conditions,s,selected['numeric'])
            query=[];components=[];ptr=[0]
            for row,i in enumerate(arguments.tolist()):
                parameter=int(s['argument_parameter'][i]);width=self.parameters_catalog[parameter].size or 1
                query.extend([row]*width);components.extend(self.numeric_index[(parameter,j)] for j in range(width));ptr.append(len(query))
            value=conditions.new_empty(0)
            if query:
                role=self.numeric_projection(F.one_hot(_ints(components,arguments),len(self.numeric_components)).float())
                value=self.numeric_decoder(base[_ints(query,arguments)]+role).flatten()
            numeric={'argument':arguments,'value':value,'ptr':_ints(ptr,arguments)}
            arguments,base=self._parameter_conditions(conditions,s,selected['reference_set'])
            optional=[];required=[];ptr=[0];required_ptr=[0];scores=[]
            for row,i in enumerate(arguments.tolist()):
                start,end=s['argument_domain_ptr'][i:i+2].tolist();domain=torch.arange(start,end,device=arguments.device)
                mask=s['argument_domain_required'][start:end];chosen=domain[~mask]
                optional.extend(chosen.tolist());required.extend(domain[mask].tolist());ptr.append(len(optional));required_ptr.append(len(required))
                scores.append(self._reference_logits(latents,s,i,base[row],chosen,self.reference_set_scorer))
            members={'argument':arguments,'logits':torch.cat(scores) if scores else conditions.new_empty(0),
                'ptr':_ints(ptr,arguments),'domain_index':_ints(optional,arguments),
                'required_ptr':_ints(required_ptr,arguments),'required_domain_index':_ints(required,arguments)}
            arguments,base=self._parameter_conditions(conditions,s,selected['reference'])
            domain_indices=[];ptr=[0];scores=[]
            for row,i in enumerate(arguments.tolist()):
                start,end=s['argument_domain_ptr'][i:i+2].tolist();domain=torch.arange(start,end,device=arguments.device)
                domain_indices.extend(domain.tolist());ptr.append(len(domain_indices))
                if self.reference_scorer is None:raise ValueError('A catalog-bound reference cannot become an unbound prediction')
                scores.append(self._reference_logits(latents,s,i,base[row],domain,self.reference_scorer))
            references={'argument':arguments,'logits':torch.cat(scores) if scores else conditions.new_empty(0),
                'ptr':_ints(ptr,arguments),'domain_index':_ints(domain_indices,arguments)}
            arguments,base=self._parameter_conditions(conditions,s,selected['boolean'])
            booleans={'argument':arguments,'logits':self.boolean_scorer(base).flatten() if len(arguments) else conditions.new_empty(0)}
            arguments,base=self._parameter_conditions(conditions,s,selected['category'])
            queries=[];options=[];ptr=[0]
            for row,i in enumerate(arguments.tolist()):
                parameter=int(s['argument_parameter'][i]);width=len(self.parameters_catalog[parameter].categories)
                queries.extend([row]*width);options.extend(self.argument_category_index[(parameter,j)] for j in range(width));ptr.append(len(queries))
            logits=conditions.new_empty(0)
            if queries:
                role=self.argument_category_projection(F.one_hot(_ints(options,arguments),len(self.argument_categories)).float())
                logits=self.argument_category_scorer(base[_ints(queries,arguments)]+role).flatten()
            categories={'argument':arguments,'ptr':_ints(ptr,arguments),'logits':logits}
            return {'numeric':numeric,'reference_set':members,'reference':references,'boolean':booleans,'category':categories}
