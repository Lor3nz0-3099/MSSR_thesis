# MSSR MAPPO Vertical Slice Implementation Plan

> **For agentic workers:** implement task-by-task with TDD.

**Goal:** ottenere un primo ciclo MARL funzionante:
8 moduli disassemblati -> Expert assembly rc_car8 -> handoff -> locomozione flat -> shared H1/H2 -> critic globale -> MAPPO -> reset.

**Architecture:** Isaac Lab gestisce simulazione/vectorization; skrl mantiene trainer e matematica MAPPO; `MSSRMAPPO` coordina actor condiviso, critic globale e optimizer separati. Le azioni passano sempre dallo stack executor MSSR esistente.

**Tech Stack:** Isaac Lab v3.0.0-beta2, Isaac Sim 6.0.0.1, skrl 2.1.0, PyTorch 2.10.0+cu128.

## Global Constraints

- Un modulo = un agente concettuale.
- Un solo H1/H2 condiviso su grafo globale actor-visible.
- Candidate raggruppate probabilisticamente per `owner_agent_id`.
- Un solo critic graph-based privilegiato e globale.
- Reward/advantage team-based.
- Actor e critic con optimizer separati.
- BC-v4 read-only come warm start.
- Nessun comando diretto ai joint: usare gli executor MSSR esistenti.
- Expert assembly iniziale escluso dalle transizioni PPO.
- N moduli = N agenti, con N variabile per costruzione; N=8 è soltanto il primo caso operativo di training.
- TDD: test RED prima di ogni implementazione.

## Tasks

- [x] Installare `skrl==2.1.0` senza alterare lo stack validato.
- [x] Verificare MAPPO ufficiale Cart-Double-Pendulum end-to-end.
- [x] Creare package minimo `drl/MARL/`.
- [ ] TDD: shared actor/critic identity e skeleton `MSSRMAPPO`.
- [ ] TDD: un forward globale con output segmentato per owner.
- [ ] TDD: actor optimizer unico + critic optimizer unico e update coordinato.
- [ ] Integrare H1/H2 ActionSet-v4 e strict warm-start BC-v4.
- [ ] Aggiungere distribuzione stocastica presence/payload, mask, logprob ed entropy per owner.
- [ ] Integrare critic GNN privilegiato.
- [ ] Integrare ActionSet -> existing executor stack.
- [ ] Integrare Expert assembly -> barriera di handoff -> MARL.
- [ ] Collegare `DirectMARLEnv`, rollout, reward/done e reset.
- [ ] Validare un ciclo completo collect -> MAPPO update -> checkpoint -> reset.

## First testable milestone

Prima di coinvolgere Isaac/SMORES, verificare in puro PyTorch/skrl che un numero arbitrario N di UID:
1. referenzi lo stesso actor object condiviso;
2. referenzi lo stesso critic object condiviso;
3. non richieda N copie delle reti;
4. funzioni almeno per N=1, N=3, N=8 e N=11, mantenendo N=8 come primo caso operativo.

I test successivi fisseranno optimizer e update condivisi prima dell'integrazione fisica.
