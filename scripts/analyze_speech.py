#!/usr/bin/env python3
"""
Analyze a speech transcript with an LLM coaching prompt and append a row
to a tracking CSV.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/analyze_speech.py path/to/transcript.txt \\
        --title "How Snakes Eat" \\
        --english-level "Advanced" \\
        --csv analysis.csv

Each run appends one row to the CSV (created with a header if it doesn't
exist yet). The full markdown feedback from the model is also saved next
to the transcript as "<transcript_stem>.feedback.md" for reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

CSV_FIELDS = [
    "presenter", "title", "source_file", "word_count",
    "english_score", "storytelling_score",
    "hook_rating", "pacing_rating", "emotional_rating", "avg_sentence_length",
    "filler_um", "filler_like", "filler_youknow", "filler_rate",
    "priority_1", "priority_2", "priority_3", "overall_impression",
]

COACH_PROMPT_TEMPLATE = """Role & Objective
Act as an expert English speaking coach and narrative strategist. Analyze the transcript of my spoken presentation (auto-transcribed, so ignore obvious minor voice-to-text typos unless they alter meaning). My current English level is [{english_level}].

Goal
Provide detailed, candid, and quote-specific feedback on both my English language mechanics and my storytelling/narrative delivery, and score each dimension independently so I can track progress over time.

Please format your response strictly using the following structure:

1. Scores
Give two scores out of 100, evaluated independently — a low score on one should not drag down the other.

- English Score: [X/100] — grammar accuracy, vocabulary precision, sentence fluency, filler-word rate. Do not factor in how good the story itself was.
- Storytelling Score: [X/100] — hook strength, pacing, structure, emotional resonance. Do not penalize for grammar mistakes if the story structure itself is strong.
One sentence justifying each score.

2. Overall Impression
A 2-3 sentence high-level summary assessing my clarity, overall structure, and conversational fluency.

3. Grammar & Phrasing Corrections
Select the 8-10 most instructive errors (not an exhaustive catalogue). For each correction, format as follows:
Original: "[Direct quote from transcript — must be copied verbatim, never paraphrased or invented]"
Correction: "[Corrected / more natural version]"
Why: [1-sentence explanation of the grammar rule or natural usage]

4. Storytelling & Narrative Impact
- The Hook & Opening: Did I capture attention immediately? How can the entry be stronger?
- Pacing & Structure (macro-level story beats): Is there a clear beginning, middle, and end? Where did the story drag or rush?
- Emotional Resonance & Details: Did I use show-don't-tell? Where can I add vivid details or stronger contrast?

5. Filler Words & Hesitations
Identify specific patterns (e.g., um, like, you know, word repetitions). Suggest concrete pacing strategies or pause techniques to replace them.

6. Vocabulary Upgrades
Provide 3-5 specific instances where a basic word can be replaced with a more precise, natural, or impactful alternative.
Original Word/Phrase: "[Quote]" -> Upgrade: "[Better Alternative]" — [Brief explanation]

7. Sentence Variety & Flow (micro-level rhythm, distinct from section 4's story pacing)
Analyze my sentence rhythm: Are the sentences too choppy, repetitive, or run-on? Suggest where to combine or split sentences for stronger spoken rhythm.

8. Top 3 Priority Focus Areas
List the 3 most impactful things I should work on for my next speech (ranked in order of priority).

Constraints
- Be direct and candid — do not soften criticism to be polite.
- Every quoted "Original" must be verbatim from the transcript.
- If the transcript is under ~150 words, skip section 4 and note that there wasn't enough material for pacing/structure analysis.

After writing the full response above, append one more section at the very end, exactly titled "9. CSV Data", containing ONLY a fenced ```json code block with this exact schema (integers where noted, no comments, no trailing text):

```json
{{
  "title": "<a concise, descriptive title for this talk, at most 4 words>",
  "english_score": <int 0-100>,
  "storytelling_score": <int 0-100>,
  "hook_rating": <int 1-10>,
  "pacing_rating": <int 1-10>,
  "emotional_rating": <int 1-10>,
  "priority_1": "<short phrase>",
  "priority_2": "<short phrase>",
  "priority_3": "<short phrase>",
  "overall_impression": "<the 2-3 sentence summary from section 2, as a single line>"
}}
```

Transcript:
\"\"\"
{transcript}
\"\"\"
"""

TIMESTAMP_RE = re.compile(r"\[[^\]]+\]\s+(\d{1,2}):(\d{2}):(\d{2})")


def strip_speaker_lines(raw: str) -> str:
    """Remove '[Speaker] HH:MM:SS' header lines, keep only spoken text."""
    lines = []
    for line in raw.splitlines():
        if TIMESTAMP_RE.match(line.strip()):
            continue
        if line.strip():
            lines.append(line.strip())
    return " ".join(lines)



def compute_filler_counts(text: str) -> dict:
    lower = text.lower()
    filler_um = len(re.findall(r"\b(um+|uh+)\b", lower))
    filler_like = len(re.findall(r"\blike\b", lower))
    filler_youknow = len(re.findall(r"\byou know\b", lower))
    word_count = len(lower.split())
    total_fillers = filler_um + filler_like + filler_youknow
    filler_rate = round((total_fillers / word_count) * 100, 2) if word_count else 0.0
    return {
        "filler_um": filler_um,
        "filler_like": filler_like,
        "filler_youknow": filler_youknow,
        "filler_rate": filler_rate,
    }


def infer_presenter(stem: str) -> str:
    """Derive presenter name (e.g. gugu/xmmz) from a filename stem like 'xmmz0906'."""
    match = re.match(r"[a-zA-Z]+", stem)
    return match.group().lower() if match else stem.lower()


def compute_avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return round(sum(lengths) / len(lengths), 2)


def call_llm(prompt: str, model: str) -> str:
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("Missing dependency: run 'pip install anthropic' first.")
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def extract_json_block(response_text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not match:
        sys.exit("Could not find a ```json``` block in the model's response.")
    return json.loads(match.group(1))


def upsert_csv_row(csv_path: Path, row: dict) -> None:
    """Append row, replacing any existing row with the same (presenter, title)."""
    existing_rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = [
                r for r in reader
                if not (r.get("presenter") == row["presenter"] and r.get("title") == row["title"])
            ]
    existing_rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt/.md file")
    parser.add_argument("--title", default=None, help="Title of the talk (default: filename)")
    parser.add_argument("--presenter", default=None, help="Presenter name, e.g. gugu/xmmz (default: inferred from filename)")
    parser.add_argument("--english-level", default="Advanced", help="Your current English level (used to tailor the coaching prompt)")
    parser.add_argument("--csv", type=Path, default=Path("analysis.csv"), help="Path to the tracking CSV")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Anthropic model to use")
    args = parser.parse_args()

    raw = args.transcript.read_text(encoding="utf-8")
    spoken_text = strip_speaker_lines(raw)
    word_count = len(spoken_text.split())
    avg_sentence_length = compute_avg_sentence_length(spoken_text)
    fillers = compute_filler_counts(spoken_text)

    prompt = COACH_PROMPT_TEMPLATE.format(english_level=args.english_level, transcript=spoken_text)
    response_text = call_llm(prompt, args.model)

    feedback_path = args.transcript.with_suffix("")
    feedback_path = feedback_path.parent / f"{feedback_path.name}.feedback.md"
    feedback_path.write_text(response_text, encoding="utf-8")

    llm_data = extract_json_block(response_text)

    row = {
        "presenter": args.presenter or infer_presenter(args.transcript.stem),
        "title": args.title or llm_data.get("title") or args.transcript.stem,
        "source_file": args.transcript.name,
        "word_count": word_count,
        "english_score": llm_data["english_score"],
        "storytelling_score": llm_data["storytelling_score"],
        "hook_rating": llm_data["hook_rating"],
        "pacing_rating": llm_data["pacing_rating"],
        "emotional_rating": llm_data["emotional_rating"],
        "avg_sentence_length": avg_sentence_length,
        "filler_um": fillers["filler_um"],
        "filler_like": fillers["filler_like"],
        "filler_youknow": fillers["filler_youknow"],
        "filler_rate": fillers["filler_rate"],
        "priority_1": llm_data["priority_1"],
        "priority_2": llm_data["priority_2"],
        "priority_3": llm_data["priority_3"],
        "overall_impression": llm_data["overall_impression"],
    }

    upsert_csv_row(args.csv, row)
    print(f"Saved row for {row['presenter']} / {row['title']} to {args.csv}")
    print(f"Full feedback saved to {feedback_path}")


if __name__ == "__main__":
    main()
