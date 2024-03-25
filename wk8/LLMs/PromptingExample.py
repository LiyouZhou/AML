# Program extended from https://huggingface.co/docs/transformers/main/tasks/prompting
# This other link provides technical information of the models available: https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5#overview
# Dependencies:
# pip install transformers torch

from transformers import pipeline
import torch

torch.manual_seed(0)
text2text_generator = pipeline("text2text-generation", model = 'google/flan-t5-large')
prompts = ["Do the following two sentences match in meaning? Sentence 1: 'a girl in a dress'. Sentence 2: 'a cat on the mat'", "Consider the following sentences. Sentence 1: 'a girl in a dress'. Sentence 2: 'a small person in a pink dress'. Are the meanings of those sentences equivalent?"]

for prompt in prompts:
    response = text2text_generator(prompt)
    print("PROMPT:%s RESPONSE:%s" % (prompt, response))
    print("---")