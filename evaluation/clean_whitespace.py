import re

pred_gpt_refined = []
with open("evaluation/gpt_transcriptions/gpt_rightquot_replaced.txt", "r", encoding="utf-8") as f:
    for line in f:
        pred_gpt_refined.append(line)

print(len(pred_gpt_refined))

#replace single quotes followed by one or more whitespaces with a single quote
result = [re.sub(r"'\s+", "'", string) for string in pred_gpt_refined]

print("Prediction after whitespace removal:", result)

with open("evaluation/gpt_transcriptions/gpt_cleaned.txt", "w", encoding="utf-8") as output:
    for line in result:
        output.write(line)