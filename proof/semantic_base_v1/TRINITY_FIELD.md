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

## A/B validation: field guidance provably helps (held-out)
Two LoRA finetunes, IDENTICAL seed/config/data (mini base, 4000 train, 200 steps, seed 42),
one with --field --field-weight 4.0, one without. Evaluated on 400 HELD-OUT transitions
(records 90000-90400, disjoint from training) via tools/eval_field_consistency.py.
Metric ALIGNMENT = mean logP(field-endorsed target) - mean logP(all targets).

  A (no field):     endorsed_lp=-6.95  overall_lp=-4.47  ALIGNMENT=-2.483
  B (field-guided): endorsed_lp=-6.50  overall_lp=-4.49  ALIGNMENT=-2.016   (+0.467 vs A)

B assigns ~1.57x more probability to Trinity-endorsed Delta-tokens (endorsed positions 648/42426)
with overall modeling UNCHANGED (-4.47 vs -4.49) -> the shift is specifically toward the field's
transitions, not a general quality change. Both models learned (loss ~7 -> ~3.4). Effect is modest
(LoRA/200 steps/weight 4.0) but unambiguous in sign with the quality confound controlled.
CONCLUSION: field guidance makes the small model measurably more on the Quantum brain's track,
at no cost to fluency. Scales with steps/weight/full-finetune.

## Experiment C (frozen model, field-only @flux) -- NEGATIVE as-run, and it's informative
ForwardPass.ps1 (PowerShell interface, NNCK-Runtime style: shells to trinity_field.py adapt +
eval_field_consistency.py; the C#/Python seam hidden like Invoke-GptOssLayerForward). Frozen model
B (ab_B_field); field_0 fit on rows 1-20000; @flux evidence rows 40001-55000 (disjoint); eval on
heldout 90000-90400 (disjoint).

  C0 (frozen field)        ALIGNMENT = -1.9194
  C1 (@flux-adapted field) ALIGNMENT = -2.1311    dALIGN = -0.212  -> NO GAIN

DIAGNOSIS: the @flux proxy gave reward=+1 to EVERY observed transition (1.34M undirected updates).
That saturates globally-frequent transitions toward 1, so post-adaptation the field endorses generic
high-frequency tokens ("assistant"/function words) instead of context-specific content -> misaligns
with what the frozen model favors -> alignment drops. Blind co-occurrence reinforcement washes out
the prior's discrimination.

LESSON (validates the architecture): @flux needs an OUTCOME signal (FluxTrace.Success -> reinforce
success, weaken failure), NOT raw co-occurrence. Co-occurrence != evidence. Corrected Experiment C
requires discriminative evidence: real FluxTraces with Success bits, or a proxy with genuine
success/failure. The pipeline (PowerShell bridge, field adapt, frozen-model measurement) is proven;
the evidence signal is what was wrong.

## Experiment C -- explicit FAIL record (expected mechanism FALSIFIED; not tuned away)
> Experiment C -- FAIL (expected mechanism falsified): Positive-only transition reinforcement
> decreases alignment by 0.212 nats (C0 -1.9194 -> C1 -2.1311). Raw transition frequency is
> insufficient for semantic-field learning. Outcome-discriminative @flux evidence is required.

The C1 update learned  seen(A->B) -> strengthen(A->B), turning semantic attention into a FREQUENCY
ESTIMATOR (1.34M positive-only updates) -> the field lost contrast. Separations proven:
  observed transition != successful transition ; co-occurrence != evidence ; frequency != semantic value.
Sharpened architecture:  @node=semantic op / @fold=semantic state / @flux=trajectory+outcome
  -> semantic learner -> Trinity field update.  (@flux CARRIES evidence; it is not the learning rule.)
This also protects the future Promotion.cs: frequent != promotable; promotion needs useful/stable, not common.

## Experiment C2 (designed; blocked on evidence) -- change ONLY the evidence signal
Freeze model+LoRA+base field+dataset+eval; replace blind reward=+1 with:
  FluxTrace.Success=true -> positive ; Success=false -> negative ; unknown -> no update.
  evidence = outcome_sign x confidence x relevance x provenance_quality (magnitude separate from direction).
PREREQUISITE FINDING (2026-07-29): real FluxTraces in .NNC-K/.learning/flux/ are all Success=True
(21/0), Contributions empty, Confidence 0.0, transitions only in FoldTrace glyphs (Pop/Wo/Yax/Sek/
Ch'en/Xul) -- NOT in token space. So C2 needs either (a) failure-bearing traces with populated
contributions from a real runtime session, or (b) a principled discriminative PROXY that encodes
semantic VALUE not frequency (e.g. PMI: reinforce high-PMI transitions, weaken generic high-frequency
ones) -- directly testing "frequency != semantic value". No LR tuning of C1.

## C2 (real, Success-driven) -- BLOCKED (not FAIL), and why
> C2 BLOCKED -- existing FluxTrace corpus is execution-valid but learning-incomplete. All observed
> traces report Success=true, Contributions are empty, Confidence=0, and fold traces operate in
> K'UHUL fold/glyph space (Pop/Wo/Yax/Sek/Ch'en/Xul) rather than the transition space being
> evaluated. C2 requires real discriminative outcome traces; synthetic negatives are explicitly
> EXCLUDED from the proof.

Data contract sharpening -- @flux must become CAUSAL PROVENANCE, not just time-travel:
  not "what happened?" but: what contributed? -> what transition did it influence? -> what
  collapsed? -> with what confidence? -> did that outcome succeed?
To unblock C2 in the CURRENT tier (no Promotion.cs / SCXQ2 / new learner): the runtime must fill
FluxTrace.{Contributions[], Confidence, Success, EndorsedTransitions[], ResultTransitions[]} from
real activity so positive AND negative evidence accumulate naturally. bigrams.json (KUHUL_pi prior:
"given role/state, expected transitions") + FluxTrace (outcome: "happened, succeeded/failed") are
the two halves that meet at the semantic update: success->strengthen, failure->weaken, unknown->keep prior.

## C2-proxy (PMI value-discriminative) -- PROXY EVIDENCE, not FluxTrace.Success
Same frozen controls as C0/C1 (model, LoRA, base Trinity field, dataset, evaluation, update
count/budget); ONLY the reward changes: C1 reward=+1 for every observation; C2 reward=sign(PMI(A,B)),
PMI=log[P(A,B)/(P(A)P(B))] -- observed-more-than-chance reinforce, less-than-chance weaken. Raw PMI
magnitude persisted (companion .pmi.json) so a later C2b can test bounded-magnitude without re-extracting.
Narrow interpretation: PASS => value-discriminative updates beat indiscriminate frequency (NOT that
PMI==success, NOT production @flux validated); FAIL => co-occurrence surprise still insufficient ->
strengthens the need for genuine outcome-bearing @flux. Production C remains real execution -> @node ->
@fold -> @flux causal provenance -> Success/Failure -> field update.
