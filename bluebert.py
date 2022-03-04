from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import torch

# import nltk
# nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree


class Annotation:
    def __init__(self, entity, tag=""):
        self.entity = entity
        self.tag = tag


id2label = {0: 'O', 1: 'B-disease', 2: 'I-disease'}

model_id = "data/clinical_ner_bluebert_pubmed_mimic_uncased"


async def predict_ner_bluebert(sentence):
    sentence = sentence.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")
    encode = tokenizer.tokenize(sentence)

    flat_pred = []
    with torch.no_grad():
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        flat_pred.extend(np.argmax(logits, axis=-1).flatten())

    predictions = [id2label[id] for id in flat_pred if id != -100]

    tags = predictions[1:len(predictions) - 1]
    pos_tags = [pos for token, pos in pos_tag(encode)]
    conlltags = [(token, pos, tg) for token, pos, tg in zip(encode, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)
    annotated_text = []
    for subtree in ne_tree:
        # skipping 'O' tags
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_text = " ".join([token for token, pos in subtree.leaves()])
            start_words = [word for word in original_text.split(" ") if "#" not in word]
            entities = []
            if len(start_words) <= 2:
                if len(start_words) == 2:
                    start = original_text.index(start_words[0])
                    end = original_text.index(start_words[1])
                    entities.append(original_text[start:end].replace(" ##", ""))
                    entities.append(original_text[end:].replace(" ##", ""))
                else:
                    entities = [entity for entity in original_text.split("##") if len(entity) != 0]
                original_text = "".join([entity for entity in entities])
                annotated_text.append(
                    Annotation(original_text, original_label)
                )

    finalTokens = []
    tokens = []
    if len(annotated_text) != 0:
        for annotation in annotated_text:
            tokens = sentence.split(annotation.entity, 1)
            if len(tokens) == 2:
                finalTokens.append(Annotation(tokens[0]))
                finalTokens.append(annotation)
                if len(tokens[1]) != 0:
                    sentence = tokens[1]
                else:
                    if annotated_text.index(annotation) == len(annotated_text) - 1:
                        sentence = ""
        if len(sentence) != 0:
            finalTokens.append(Annotation(sentence))

    return finalTokens
