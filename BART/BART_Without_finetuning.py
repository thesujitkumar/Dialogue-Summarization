import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import sacrebleu
import torch

df= pd.read_csv("clean_test.csv")
# df=df.head(10)
news_texts = df['dialogue']
news_list = list(news_texts)

generated_summary = []

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

sum=[]
for a in news_list:
    input_ids = tokenizer(a, return_tensors="pt", max_length=1024, padding=True,truncation=True).to(device)
    summary_ids = model.generate(input_ids['input_ids'], max_length=30, min_length = 10, num_beams=4, length_penalty=2.0, early_stopping=True)
    generated_headlines = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    print(generated_headlines)
    sum.extend(generated_headlines)
reference = df['summary']
ref_list = list(reference)
# for i in sum:
#     print(type(i))

print( " the number of sample in generated summary is", len(sum))
print( " the number of sample in generareference summary is", len(ref_list))
#
# bleu = sacrebleu.corpus_bleu(sum, [ref_list])
# print("BLEU Score:", bleu.score)

bleu = sacrebleu.corpus_bleu(sum, [ref_list])
normalized_bleu = bleu.score / 100.0
print("Normalized BLEU Score:", normalized_bleu)
