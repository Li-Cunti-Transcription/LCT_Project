import pywer

references = []
with open("evaluation/txt/ground_truth.txt", "r", encoding="utf-8") as f:
    for line in f:
        references.append(line)

print(references)
print(len(references))

pred_transkribus = []
with open("evaluation/transkribus_transcriptions/transkribus.txt", "r", encoding="utf-8") as f:
    for line in f:
        pred_transkribus.append(line)

print(pred_transkribus)
print(len(pred_transkribus))

pred_gpt = []
with open("evaluation/gpt_transcriptions/gpt.txt", "r", encoding="utf-8") as f:
    for line in f:
        pred_gpt.append(line)

print(pred_gpt)
print(len(pred_gpt))


wer_transkribus = pywer.wer(references, pred_transkribus)
cer_transkribus = pywer.cer(references, pred_transkribus)
print("Transkribus overall WER and CER:")
print(f"WER: {wer_transkribus:.2f}, CER: {cer_transkribus:.2f}")
wer_gpt = pywer.wer(references, pred_gpt)
cer_gpt = pywer.cer(references, pred_gpt)
print("GPT overall WER and CER:")
print(f"WER: {wer_gpt:.2f}, CER: {cer_gpt:.2f}")


""" mapping = {
    "’": "'",
    "í": "i",
    "é": "è"
}

def replace_char(list_of_strings, mapping_dict):
    # initialize empty list
    updated_list = []
    # for each line in the input list...
    for line in list_of_strings:
        # initialize an empty list...
        updated_line = []
        # split the line at whitespace and iterate over the words...
        for word in line.split():
            # iterate over each character (= for char in word), check for the inclusion of the key <char> in the input dictionary and return <char> as default value if the key doesn't exist (= mapping_dict.get(char, char)), then join the characters
            updated_word = ''.join(mapping_dict.get(char, char) for char in word)
            # append the word to the line
            updated_line.append(updated_word)
        # append the line to the list
        updated_list.append(' '.join(updated_line))
    
    return updated_list

pred_corrected = replace_char(pred_gpt, mapping)

wer_gpt = pywer.wer(references, pred_corrected)
cer_gpt = pywer.cer(references, pred_corrected)
print("GPT overall WER and CER after character replacement:")
print(f"WER: {wer_gpt:.2f}, CER: {cer_gpt:.2f}") """

