# Slice A: Trinity weighted semantic field (fit + @flux inline update)

Makes Trinity "better not bigger": fits its weighted field to observed dataset structure
(not more ELIZA rules), with the @flux inline-learning rule. ELIZA/RegEx/Roslyn keep explicit
recognition; this carries the learnable geometry of meaning.

tools/trinity_field.py:
  fit  <transitions.jsonl> <field.json>   -- bootstrap the PRIOR from F(t+1)=Preserve+Delta data
  demo <field.json>                       -- show the @flux inline update

Learns: op_given_lane P(op|lane), delta_given_op P(new|op), transition[p][d] (Preserve->Delta
semantic bigram), pd_stats (mean Preserve/Delta per op). @flux(p,d,reward) tunes a transition
toward 1 on positive evidence, toward 0 on negative (bounded [0,1]).

## Result (30k transitions, vocab 2000, 1967 transition nodes, 1.2MB field.json)
P(op|lane): code->INSTRUCT 0.28 (dominant), math->GENERATE 0.17 elevated, general->RESPOND/DEFINE.
Preserve/Delta geometry: COMPARE delta=22.1 (most new), DEFINE preserve=5.7/delta=19.7 (compact).
@flux demo 'human'->'assistant': prior 0.067 -> +evidence 0.114/0.158/0.200/0.240 -> -evidence 0.228/0.217.

## Three weight tiers (kept separate, per the architecture)
prior (this fit) / inline (flux() runtime) / persistent (future SCXQ2 promotion).
Next: use guidance() to steer the small-model finetune; wire @flux to real execution evidence;
port weights into quantum_trinity_hybrid.cpp; promote stable transitions to XSQ2 xshards.

## Field-guided finetune (the loop closed)
tools/finetune_hf.py --field <field.json> [--field-weight W]: for each example, Trinity's
guidance(preserve) picks the endorsed Delta-concept token ids; the per-token CE is upweighted (xW)
where the target token is field-endorsed, so the small model prioritizes field-consistent
transitions. Composes with --lora (both levers in one command). Verified end-to-end: field loads,
endorsed sets built per example, weighted CE trains, save_servable -> GGUF path intact.
This is "the field guides the tokens" operational in training.
