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



# calculate diffing
def compute_diff(references_list, predicted_list, diff_file="evaluation/diff_file.txt"):
    with open(diff_file, "w", encoding="utf-8") as f:

        for i in range(len(references_list)):
            f.write(f"Processing line {i + 1}...\n")
            ref_line = references_list[i]
            pred_line = predicted_list[i]
            f.write(f"Reference: {ref_line}\n")
            f.write(f"Prediction: {pred_line}\n")
            diff = difflib.ndiff(ref_line.split(), pred_line.split())
            # casting the generator into a list
            diff_list = list(diff)
            f.write("\n".join(diff_list))
            f.write("\n" + "="*50 + "\n")

#compute_diff(references, predictions)

html_diff = difflib.HtmlDiff()

# HTML difference table
html_content = html_diff.make_table(references, predictions, fromdesc="Ground Truth Text", todesc="GPT Transcription")

# Save the result to an HTML file
with open("evaluation/diff_table.html", "w", encoding="utf-8") as file:
    file.write(html_content)