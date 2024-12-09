import matplotlib
import matplotlib.pyplot as plt

# Example data
models = ['Basile Printed_fine-tuned', 'ChatGPT']
wer = [2.44, 24.65]  # WER values
cer = [0.51, 5.45]  # CER values

# Plotting
x = range(len(models))
plt.bar(x, wer, width=0.4, label='WER', align='center')
plt.bar([p + 0.4 for p in x], cer, width=0.4, label='CER', align='center')
plt.xticks([p + 0.2 for p in x], models)
plt.xlabel('Models')
plt.ylabel('Error Rate')
plt.title('WER and CER Comparison')
plt.legend()
plt.show()
