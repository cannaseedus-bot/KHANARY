# Semantic base -- slice #1 + #1b: clean corpus -> Preserve(+)Delta transition dataset

Turns flat instruction/chat data into SEMANTIC-PATTERN (transition-shaped) training examples that
teach the transition function F(t+1)=Preserve(+)Delta (the stack's semantic-tensor target), not recall.

## Pipeline
1. tools/clean_corpus.py  -> E:\data\khanary_clean_train.jsonl : 101,727 pairs from 6 raw JSONL
   (help/instruct/qa + ChatGPT/DeepSeek exports + a 143MB G-code blob). Strips images/base64/
   opcodes/G-code/disasm/control-chars so STRAY opcodes can't shadow the K'UHUL/glyph opcode space.
   VERIFIED clean: 0 image payloads, 0 base64, 0 hexdumps, 0 G-code, 0 ctrl-chars (39 residual
   'data:image' are harmless text mentions). 0032 G-code blob -> 0 pairs (correctly rejected).
2. tools/build_transitions.py -> E:\data\khanary_transitions.jsonl : 101,562 transitions
   {state, lane, op, preserve[], delta[], state_next, text}. avg Preserve 9.2 / Delta 16.3.

## Result (slice #1b -- op diversity fixed)
op coverage: RESPOND 28.6% (was 95.5% before scored classifier), INSTRUCT 18.5%, DEFINE 17.0%,
GENERATE 13.3%, EXPLAIN 8.2%, ENUMERATE 6.8%, COMPUTE 3.9%, TRANSFORM 2.2%, COMPARE 0.9%, DECLINE 0.6%.
lane coverage (expert routing signal): general 77.8% (-> layered_instruct), code 18.9%
(-> layered_code), math 3.3% (-> layered_math).

## Honest scope
- Labels are heuristic regex (ELIZA-mechanism decompose) -> routing scaffold, not ground truth.
- LOCATE effectively dead (4); math lane thin (3.3%) -> layered_math undersampled.
- This is the training SUBSTRATE; no training run performed here.
