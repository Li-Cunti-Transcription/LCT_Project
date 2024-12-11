import difflib

references = []
with open("evaluation/txt/ground_truth.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        references.append(line)


predictions = []
with open("evaluation/gpt_transcriptions/gpt.txt", "r", encoding="utf-8") as f:
    for line in f:
        predictions.append(line)


html_diff = difflib.HtmlDiff()

# HTML difference table
html_content = html_diff.make_table(references, predictions, fromdesc="Ground Truth Text", todesc="GPT Transcription")

# Save the result to an HTML file
with open("evaluation/diff_table.html", "w", encoding="utf-8") as file:
    file.write(html_content)