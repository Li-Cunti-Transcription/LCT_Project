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

pred_print_M1 = []
with open("evaluation/transkribus_transcriptions/transkribus_printm1.txt", "r", encoding="utf-8") as f:
    for line in f:
        pred_print_M1.append(line)

print(len(pred_print_M1))

# overall WER and CER
wer_transkribus = pywer.wer(references, pred_transkribus)
cer_transkribus = pywer.cer(references, pred_transkribus)
print("Transkribus overall WER and CER:")
print(f"WER: {wer_transkribus:.2f}, CER: {cer_transkribus:.2f}")
wer_gpt = pywer.wer(references, pred_gpt)
cer_gpt = pywer.cer(references, pred_gpt)
print("GPT overall WER and CER:")
print(f"WER: {wer_gpt:.2f}, CER: {cer_gpt:.2f}")
wer_transkribus_print = pywer.wer(references, pred_print_M1)
cer_transkribus_print = pywer.cer(references, pred_print_M1)
print("Transkribus Print overall WER and CER:\n", f"WER: {wer_transkribus_print:.2f}, CER: {cer_transkribus_print:.2f}")

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
tr_print_page_1, tr_print_page_2, tr_print_page_3 = split_in_pages(pred_print_M1)

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

wer_tr_print_1 = pywer.wer(ref_page_1, tr_print_page_1)
cer_tr_print_1 = pywer.cer(ref_page_1, tr_print_page_1)
print("Transkribus Print WER and CER for page 1:\n", f"WER: {wer_tr_print_1:.2f}, CER: {cer_tr_print_1:.2f}")
wer_tr_print_2 = pywer.wer(ref_page_2, tr_print_page_2)
cer_tr_print_2 = pywer.cer(ref_page_2, tr_print_page_2)
print("Transkribus Print WER and CER for page 2:\n", f"WER: {wer_tr_print_2:.2f}, CER: {cer_tr_print_2:.2f}")
wer_tr_print_3 = pywer.wer(ref_page_3, tr_print_page_3)
cer_tr_print_3 = pywer.cer(ref_page_3, tr_print_page_3)
print("Transkribus Print WER and CER for page 3:\n", f"WER: {wer_tr_print_3:.2f}, CER: {cer_tr_print_3:.2f}")

# compute scores after new prompts
pred_gpt_refined = []
with open("evaluation/gpt_transcriptions/gpt_rightquot_replaced.txt", "r", encoding="utf-8") as f:
    for line in f:
        pred_gpt_refined.append(line)

print(len(pred_gpt_refined))

wer_gpt_refined = pywer.wer(references, pred_gpt_refined)
cer_gpt_refined = pywer.cer(references, pred_gpt_refined)
print("GPT WER and CER after refined prompts:\n", f"WER: {wer_gpt_refined:.2f}, CER: {cer_gpt_refined:.2f}")

gpt_page_1_refined, gpt_page_2_refined, gpt_page_3_refined = split_in_pages(pred_gpt_refined)

# computing updated scores per page
wer_gpt_1_ref = pywer.wer(ref_page_1, gpt_page_1_refined)
cer_gpt_1_ref = pywer.cer(ref_page_1, gpt_page_1_refined)
print("GPT WER and CER for page 1 after refined prompts:\n", f"WER: {wer_gpt_1_ref:.2f}, CER: {cer_gpt_1_ref:.2f}")
wer_gpt_2_ref = pywer.wer(ref_page_2, gpt_page_2_refined)
cer_gpt_2_ref = pywer.cer(ref_page_2, gpt_page_2_refined)
print("GPT WER and CER for page 2 after refined prompts:\n", f"WER: {wer_gpt_2_ref:.2f}, CER: {cer_gpt_2_ref:.2f}")
wer_gpt_3_ref = pywer.wer(ref_page_3, gpt_page_3_refined)
cer_gpt_3_ref = pywer.cer(ref_page_3, gpt_page_3_refined)
print("GPT WER and CER for page 3 after refined prompts:\n", f"WER: {wer_gpt_3_ref:.2f}, CER: {cer_gpt_3_ref:.2f}")

# scores for page 1 after long-s prompt
pred_gpt_longs = []
with open("evaluation/gpt_transcriptions/gpt_longs_replaced_page1.txt", "r", encoding="utf-8") as f:
    for line in f:
        pred_gpt_longs.append(line)

print(len(pred_gpt_longs))

wer_gpt_longs = pywer.wer(references, pred_gpt_longs)
cer_gpt_longs = pywer.cer(references, pred_gpt_longs)
print("GPT WER and CER for page 1 after long-s correction prompts:\n", f"WER: {wer_gpt_longs:.2f}, CER: {cer_gpt_longs:.2f}")