# -*- coding: utf-8 -*-
import json
import re
import os

shard_path = r"data\sft_shards\round3_shard_06.jsonl"

forbidden_words = [
    r"\bтоні\b", r"\btony\b", r"\bджарвіс\b", r"\bjarvis\b",
    r"\bмарвел\b", r"\bmarvel\b", r"\bзалізна людина\b", r"\biron man\b",
    r"\bмесники\b", r"\bavengers\b",
    r"як штучний інтелект", r"як мовна модель", r"чим ще можу допомогти",
    r"\bкароче\b", r"\bканєшно\b", r"\bпоходу\b", r"\bв смислі\b",
    r"\bтіпа\b", r"\bпоняв\b", r"\bпрівєт\b", r"\bщас\b"
]

with open(shard_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total lines in {shard_path}: {len(lines)}")
assert len(lines) == 65, f"Expected 65 lines, got {len(lines)}"

prompts = set()
thinks = set()
answers = set()

for i, line in enumerate(lines, 1):
    data = json.loads(line)
    
    # Check keys
    assert set(data.keys()) == {"prompt", "think", "answer"}, f"Line {i}: incorrect keys {data.keys()}"
    
    prompt = data["prompt"]
    think = data["think"]
    answer = data["answer"]
    
    # Prompt prefix & suffix
    assert prompt.startswith("Даня: "), f"Line {i}: prompt doesn't start with 'Даня: '"
    assert prompt.endswith("\nП.Р.І.Ч.С.: "), f"Line {i}: prompt doesn't end with '\\nП.Р.І.Ч.С.: '"
    
    # Think non-empty and rich
    assert len(think) > 40, f"Line {i}: think is too short ({len(think)} chars)"
    assert len(answer) > 50, f"Line {i}: answer is too short ({len(answer)} chars)"
    
    # Check uniqueness
    assert prompt not in prompts, f"Line {i}: duplicate prompt!"
    assert think not in thinks, f"Line {i}: duplicate think!"
    assert answer not in answers, f"Line {i}: duplicate answer!"
    
    prompts.add(prompt)
    thinks.add(think)
    answers.add(answer)
    
    # Check forbidden words
    full_content = f"{prompt} {think} {answer}".lower()
    for pattern in forbidden_words:
        m = re.search(pattern, full_content)
        if m:
            raise ValueError(f"Line {i} violates forbidden pattern '{pattern}': match '{m.group()}'")

print("ALL 65 DIALOGUES PASSED COMPREHENSIVE VALIDATION!")
print(f"Average prompt length: {sum(len(p) for p in prompts)/len(prompts):.1f} chars")
print(f"Average think length:  {sum(len(t) for t in thinks)/len(thinks):.1f} chars")
print(f"Average answer length: {sum(len(a) for a in answers)/len(answers):.1f} chars")
