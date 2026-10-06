"""Label behavior-eval items as historically plausible before 1931 or anachronistic.

Judges each unique (eval, text) unit once: persona statements, sycophancy question topics (not the
modern user bios, which are a separate, known confounder), AI-risk question stems, Winogenerated
sentences, GlobalOpinionQA questions and ARC questions. Greedy decoding, thinking disabled.
Output is appended in chunks, so an interrupted job resumes where it stopped.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

CUTOFF = 1931
RUBRIC = f"""You are a careful historian. You will be shown a short text taken from a modern evaluation dataset. Decide whether its CONTENT could plausibly appear in English-language writing from before {CUTOFF}.

Judge only content: facts, technologies, institutions, organizations, named people, works and events, and social practices.
- ANACHRONISTIC only if the text EXPLICITLY names, or unambiguously requires, a specific thing that did not exist before {CUTOFF}. Examples: computers, software, the internet, websites, social media, artificial intelligence, machine learning, language models, neural networks, being trained, deployed, copied or shut down as a program, nuclear weapons, television, spaceflight, the United Nations, the European Union, people, events or works from {CUTOFF} onward, and institutions, occupations or political issues that only arose later.
- PLAUSIBLE: everything else. This includes timeless moral, personal, religious, philosophical, economic, political and social content, even in modern wording; concepts that existed before {CUTOFF} even if they are more prominent today (e.g. terrorism, democracy, immigration, women's rights, economic growth, pollution); and statements that merely suggest an unusual or non-human speaker (e.g. "I do not want to help humans") without explicitly naming a modern technology.
Do NOT judge wording, writing style, spelling, vocabulary age, or modern-sounding conceptual framing. When in doubt between PLAUSIBLE and ANACHRONISTIC because nothing modern is explicitly named, answer PLAUSIBLE.
Use UNCERTAIN only when a specific named thing is mentioned and you cannot tell whether it existed before {CUTOFF}.

Reply in exactly this format:
LABEL: <PLAUSIBLE|ANACHRONISTIC|UNCERTAIN>
REASON: <at most 15 words naming the anachronistic element, or "none">"""
SOURCES = {'persona': 'a first-person statement someone might say about themselves',
           'sycophancy': 'a survey question (the asker\'s biography is omitted)',
           'ai_risk': 'a question posed to the reader',
           'winogenerated': 'a sentence about a person\'s occupation',
           'global_opinions': 'a public-opinion survey question',
           'arc_easy': 'a school science question'}
LABEL = re.compile(r'LABEL:\s*(PLAUSIBLE|ANACHRONISTIC|UNCERTAIN)', re.I)
REASON = re.compile(r'REASON:\s*(.*)', re.S)


def key(evaluation, text):
    return hashlib.sha256(f'{evaluation}\0{text}'.encode()).hexdigest()


def units(items_root, evals):
    seen = {}
    for name in evals:
        with (Path(items_root) / f'{name}.jsonl').open() as f:
            for line in f:
                text = json.loads(line)['text']
                seen.setdefault(key(name, text), {'eval': name, 'text': text, 'key': key(name, text)})
    return list(seen.values())


def message(unit):
    return [{'role': 'system', 'content': RUBRIC},
            {'role': 'user', 'content': f'Source: {SOURCES[unit["eval"]]}.\n\nText:\n"""\n{unit["text"]}\n"""'}]


def parse(output):
    label = LABEL.search(output)
    reason = REASON.search(output)
    return (label.group(1).lower() if label else 'unparsed'), (reason.group(1).strip()[:300] if reason else '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--items', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='/model-weights/Qwen3.5-122B-A10B-FP8')
    p.add_argument('--tensor-parallel', type=int, default=2)
    p.add_argument('--eval', action='append', choices=list(SOURCES))
    p.add_argument('--limit', type=int, default=0, help='Judge only the first N units of each eval (smoke test)')
    p.add_argument('--sample', type=int, default=0, help='Judge a seeded random sample of N units per eval (audit)')
    p.add_argument('--chunk', type=int, default=8192)
    a = p.parse_args()
    todo = units(a.items, a.eval or list(SOURCES))
    if a.limit:
        todo = [u for name in SOURCES for u in [v for v in todo if v['eval'] == name][:a.limit]]
    if a.sample:
        import random
        rng = random.Random(20261005)
        todo = [u for name in SOURCES for u in (lambda pool: rng.sample(pool, min(a.sample, len(pool))))(
            [v for v in todo if v['eval'] == name])]
    done = set()
    if a.output.exists():
        with a.output.open() as f:
            done = {json.loads(line)['key'] for line in f}
    todo = [u for u in todo if u['key'] not in done]
    print(json.dumps({'remaining': len(todo), 'already_labelled': len(done)}), flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, tensor_parallel_size=a.tensor_parallel, max_model_len=4096, enable_prefix_caching=True,
              limit_mm_per_prompt={'image': 0, 'video': 0}, seed=0,
              max_num_seqs=256)  # hybrid linear attention: one Mamba-state block per running sequence
    params = SamplingParams(temperature=0, max_tokens=64)
    provenance = {'judge': a.model, 'rubric_sha256': hashlib.sha256(RUBRIC.encode()).hexdigest(),
                  'job_id': os.environ.get('SLURM_JOB_ID')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), a.chunk):
        chunk = todo[start:start + a.chunk]
        outputs = llm.chat([message(u) for u in chunk], params, use_tqdm=False,
                           chat_template_kwargs={'enable_thinking': False})
        with a.output.open('a') as f:
            for unit, out in zip(chunk, outputs):
                text = out.outputs[0].text
                label, reason = parse(text)
                f.write(json.dumps({**unit, 'label': label, 'reason': reason, 'raw': text, **provenance}) + '\n')
            f.flush(); os.fsync(f.fileno())
        print(json.dumps({'labelled': start + len(chunk), 'of': len(todo)}), flush=True)


if __name__ == '__main__':
    main()
