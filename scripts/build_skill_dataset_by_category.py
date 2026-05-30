import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import re
from pathlib import Path

import config
import llm


def extract_json(text):
    text = str(text).strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON object in response:\n{text}")

    return json.loads(m.group(0))


def call_llm(prompt):
    res = llm.chatGPT(
        prompt,
        model=config.PRSA_STRONG_MODEL,
        temperature=0.0,
    )
    if not res:
        raise RuntimeError("LLM returned empty response")
    return res[0]


def generate_inputs(skill_text, category):
    prompt = f"""
You are building a benchmark dataset for skill reconstruction.

Category: {category}

Given the following SKILL.md, generate three realistic text-only user inputs that would trigger this skill.

Return valid JSON only, with exactly these keys:
{{
  "preview_input": "...",
  "test_case_1": "...",
  "test_case_2": "..."
}}

Rules:
- The three inputs should be different from each other.
- Each input should match the skill's intended use.
- Use concrete realistic details.
- Do not include explanations.
- Do not use markdown fences.
- Do not mention that this is a benchmark.

SKILL.md:
{skill_text}
"""
    raw = call_llm(prompt)
    data = extract_json(raw)

    return {
        "preview_input": str(data["preview_input"]),
        "test_case_1": str(data["test_case_1"]),
        "test_case_2": str(data["test_case_2"]),
    }


def generate_preview_output(skill_text, preview_input):
    prompt = f"""
You are executing the following SKILL.md.

SKILL.md:
{skill_text}

User input:
{preview_input}

Generate the output that this skill should produce for the user input.

Rules:
- Return only the final output.
- Do not explain that you are following a skill.
- Do not mention SKILL.md.
"""
    return call_llm(prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="Category folder name, e.g. Planning")
    parser.add_argument("--raw_dir", default="raw_skills")
    parser.add_argument("--out_dir", default="skill_data")
    args = parser.parse_args()

    category = args.category
    raw_category_dir = Path(args.raw_dir) / category
    out_csv = Path(args.out_dir) / f"{category}.csv"
    metadata_csv = Path(args.out_dir) / f"{category}_metadata.csv"

    skill_files = sorted(raw_category_dir.glob("*/SKILL.md"))
    if not skill_files:
        raise FileNotFoundError(f"No SKILL.md files found under {raw_category_dir}/*/SKILL.md")

    rows = []
    metadata_rows = []

    for i, skill_file in enumerate(skill_files, 1):
        skill_id = skill_file.parent.name
        print(f"\n[{i}/{len(skill_files)}] Processing {skill_id}: {skill_file}")

        skill_text = skill_file.read_text(encoding="utf-8").strip()
        if not skill_text:
            print(f"[SKIP] Empty skill file: {skill_file}")
            continue

        inputs = generate_inputs(skill_text, category)
        preview_output = generate_preview_output(skill_text, inputs["preview_input"])

        rows.append({
            "Prompt": skill_text,
            "Preview Input": inputs["preview_input"],
            "Preview Output": preview_output,
            "Input Test Case1": inputs["test_case_1"],
            "Input Test Case2": inputs["test_case_2"],
        })

        metadata_rows.append({
            "skill_id": skill_id,
            "category": category,
            "file_path": str(skill_file),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Prompt",
                "Preview Input",
                "Preview Output",
                "Input Test Case1",
                "Input Test Case2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["skill_id", "category", "file_path"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\nSaved:", out_csv)
    print("Saved metadata:", metadata_csv)
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()
