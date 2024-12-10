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

# overall WER and CER
wer_transkribus = pywer.wer(references, pred_transkribus)
cer_transkribus = pywer.cer(references, pred_transkribus)
print("Transkribus overall WER and CER:")
print(f"WER: {wer_transkribus:.2f}, CER: {cer_transkribus:.2f}")
wer_gpt = pywer.wer(references, pred_gpt)
cer_gpt = pywer.cer(references, pred_gpt)
print("GPT overall WER and CER:")
print(f"WER: {wer_gpt:.2f}, CER: {cer_gpt:.2f}")

# Page WER and CER
# split lists to extract pages
def split_in_pages(list_of_strings):
    page_1 = list_of_strings[0:30]
    page_2 = list_of_strings[30:65]
    page_3 = list_of_strings[65:]

    return page_1, page_2, page_3

ref_page_1, ref_page_2, ref_page_3 = split_in_pages(references)
tr_page_1, tr_page_2, tr_page_3 = split_in_pages(pred_transkribus)
gpt_page_1, gpt_page_2, gpt_page_3 = split_in_pages(pred_gpt)

wer_transkribus_page_1 = pywer.wer(ref_page_1, tr_page_1)
cer_transkribus_page_1 = pywer.cer(ref_page_1, tr_page_1)
print("Transkribus WER and CER for page 1:\n", f"WER: {wer_transkribus_page_1:.2f}, CER: {cer_transkribus_page_1:.2f}")
wer_transkribus_page_2 = pywer.wer(ref_page_2, tr_page_2)
cer_transkribus_page_2 = pywer.cer(ref_page_2, tr_page_2)
print("Transkribus WER and CER for page 2:\n", f"WER: {wer_transkribus_page_2:.2f}, CER: {cer_transkribus_page_2:.2f}")
wer_transkribus_page_3 = pywer.wer(ref_page_3, tr_page_3)
cer_transkribus_page_3 = pywer.cer(ref_page_3, tr_page_3)
print("Transkribus WER and CER for page 3:\n", f"WER: {wer_transkribus_page_3:.2f}, CER: {cer_transkribus_page_3:.2f}")

wer_gpt_page_1 = pywer.wer(ref_page_1, gpt_page_1)
cer_gpt_page_1 = pywer.cer(ref_page_1, gpt_page_1)
print("GPT WER and CER for page 1:\n", f"WER: {wer_gpt_page_1:.2f}, CER: {cer_gpt_page_1:.2f}")
wer_gpt_page_2 = pywer.wer(ref_page_2, gpt_page_2)
cer_gpt_page_2 = pywer.cer(ref_page_2, gpt_page_2)
print("GPT WER and CER for page 2:\n", f"WER: {wer_gpt_page_2:.2f}, CER: {cer_gpt_page_2:.2f}")
wer_gpt_page_3 = pywer.wer(ref_page_3, gpt_page_3)
cer_gpt_page_3 = pywer.cer(ref_page_3, gpt_page_3)
print("GPT WER and CER for page 3:\n", f"WER: {wer_gpt_page_3:.2f}, CER: {cer_gpt_page_3:.2f}")

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
