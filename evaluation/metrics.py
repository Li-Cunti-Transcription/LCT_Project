import re
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


references = []
with open("evaluation/txt/ground_truth.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        references.append(line)

print(references)
print(len(references))

predictions = []
with open("evaluation/gpt_transcriptions/gpt.txt", "r", encoding="utf-8") as f:
    for line in f:
        predictions.append(line)

print(predictions)
print(len(predictions))

def tokenize(list_of_strings):
    tokens = []
    # iterate over lines in the list
    for string in list_of_strings:
        # for each line, remove leading and trailing spaces and splite at whitespace, then store the resulting list of words in the variable "words"
        words = string.strip().split(" ")
        # for each word in the words list, convert the string into a list and store it in the variable "chars"
        for word in words:
            chars = list(word)
            # extend the tokens list with the resulting list
            tokens.extend(chars)
    
    return tokens

# function call for reference
ref_tokens = tokenize(references)
# function call for prediction
pred_tokens = tokenize(predictions)

# check length of both lists
print("The length of reference tokens is:", len(ref_tokens))
print("The length of predicted tokens is:", len(pred_tokens))

# get the sorted set of tokens from both lists
tokens_list = sorted(set(ref_tokens + pred_tokens))

# initialize label encoder
le = LabelEncoder()
# map the tokens to numerical values
le.fit(tokens_list)

# transform tokens to labels according to the mapping performed by .fit()
ref_labels = le.transform(ref_tokens)
pred_labels = le.transform(pred_tokens)


# check both sets of tokens and their length
ref_set = set(ref_tokens)
print("Reference set length:", len(ref_set))
pred_set = set(pred_tokens)
print("Prediction set length:", len(pred_set))

""" matrix = confusion_matrix(ref_labels, pred_labels)
print("confusion matrix:\n", matrix) """

""" def align_chars(reference, prediction):
    aligned_list = []
    for line in reference:
        pass """

