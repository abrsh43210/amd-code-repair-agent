# AMD Code Repair Agent

An open-source code-repair agent that runs a local code model on an **AMD Instinct GPU (ROCm)**.
Give it a buggy Python file with tests, or a whole repository plus an issue text, and it returns
a fixed file or a `git apply`-able patch. Everything runs offline inside one Docker image built on
AMD's ROCm PyTorch base; no API keys, no network calls at run time.

Built for the AMD AI League (lablab.ai × AMD), Mini-Challenge 5 "Code Repair", and released
under the MIT license.

## Why it is built this way

For repair, **verification** is the lever. A small model's one-shot patch is a coin flip; the same
model inside a propose → verify → keep-the-best loop only ever contributes patches that were checked.

### Single-file mode
1. **Already green?** If the given tests pass on the original, return it untouched.
2. **Model-free mutation search** (~400 single edits: flipped comparisons, off-by-one bounds,
   `//` vs `/`, `and`/`or`, `min`/`max`, mutable defaults, dropped returns…), lines named in the
   traceback first, each run in its own sandboxed process. Fixes the classic one-token bugs in ~0.1 s.
3. **The model** (Qwen2.5-Coder, resident on the GPU) proposes minimal fixes; each is verified the same way.
4. **Select**: `pass > compiles > unknown > fail > broken`. An unverified patch never displaces a verified one.

### Repository mode (SWE-bench shaped)
1. **Localize** (`app/mc5/localize.py`, model-free): files are split along their AST into
   functions/classes and ranked with BM25 against the issue, boosted for symbols, paths and
   traceback frames the issue names. Time-boxed, so huge repos still answer in time.
2. **Pick + focus**: the model reads signatures-only skeletons of the top files and pins the
   symbols it wants; the picked file leads the context.
3. **Edit**: SEARCH/REPLACE blocks, applied tolerantly (re-indentation, missing blank lines,
   elided `...`, wrong file name, sync/async twins) and refused when ambiguous, with the reason
   and the closest real lines fed back for a retry. If no edit applies, the model rewrites the
   whole target function instead.
4. **Check**: the edit must compile; nearby existing tests run before and after in a time-boxed
   overlay copy. The repository on disk is never modified.

Every task finishes inside a 25 s budget (the league limit is 30 s per item).

## Results so far (honest numbers)

On 25 practice tasks rebuilt from the public history of `requests`, `httpx`, `rich` and
`fastapi` (`practice/tasks.jsonl`, each verified: tests fail before the reference fix, pass after):

| Setup | Resolved | Notes |
|---|---|---|
| Plumbing with a perfect "oracle" model | 25 / 25 | the ceiling the pipeline allows |
| Right file in the model's top 3 | 22 / 25 | model-free localization |
| Qwen2.5-Coder-14B on AMD MI300X, 25 s per task | 4 / 25 | 24/25 produce a patch, slowest task 21.8 s |

Speed on the MI300X is not the bottleneck (prompt read < 1 s, ~30 tokens/s out for 14B);
the remaining misses are choosing the right file/spot and multi-hunk fixes.

## Run it

```bash
# offline tests (no GPU needed)
python3 tests/run_local.py                 # single-file mode, model-free search
MC5_FAKE_LLM=1 python3 tests/run_local.py  # also exercises the model path with a test double
python3 tests/test_repo_mode.py            # repository mode
python3 tests/test_repo_timing.py          # time-budget guarantees

# one task, inside the image
python3 /app/app.py --input /app/input/task.json   # writes /app/output/task_output.json

# practice evaluation (needs clones of the four repos)
python3 tools/mine_practice_tasks.py CLONES practice/tasks.jsonl
python3 tests/eval_practice.py --tasks practice/tasks.jsonl --repos CLONES --mode oracle
```

Build the image with the `Dockerfile` (`linux/amd64`, base `rocm/pytorch`). The model weights are
downloaded at build time into `/models`, so the container never needs the network.

## Layout

```
app/app.py            entry point: argument shapes, output files, nothing on stdout
app/mc5/task.py       input parsing + output writing
app/mc5/repair.py     single-file propose/verify/select pipeline
app/mc5/heuristics.py model-free candidate generation
app/mc5/verify.py     sandboxed candidate verification
app/mc5/localize.py   BM25 + AST localization for repository mode
app/mc5/repo.py       repository-mode pipeline (pick, edit, rewrite, check, patch)
app/mc5/model.py      the code model on the AMD GPU (ROCm shows up as torch.cuda)
app/mc5/server.py     resident model server over a unix socket (weights load once)
tests/                offline tests and the practice evaluator
practice/             25 practice tasks + localization / oracle reports
```

## License

MIT, see [LICENSE](LICENSE).
