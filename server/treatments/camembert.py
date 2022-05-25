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


id2label = {0: "B-examen", 1: "I-mode", 2: "B-dose", 3: "I-substance", 4: "I-traitement", 5: "I-valeur",
            6: "B-substance", 7: "B-traitement", 8: "O", 9: "I-moment", 10: "B-valeur", 11: "B-mode",
            12: "I-anatomie", 13: "I-dose", 14: "B-moment", 15: "B-anatomie", 16: "I-examen"}


model_id = "server/data/deft_8_classes_camembert"


async def predict_ner_camembert(sentence):
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

    # print("predictions =", predictions)
    tags = predictions[1:len(predictions) - 1]
    # print("tags before =", tags)

    # for token, tag in zip(encode, tags):
    #    print("token =", token, "prediction =", tag)

    cpt = 0
    temp = "O"
    for code, tag in zip(encode, tags):
        if not code.startswith("▁") and tag.startswith("O") and not temp.startswith("O"):
            tags[cpt] = "I" + temp[1:]
        if not code.startswith("▁") and tag.startswith("B") and not temp.startswith("O"):
            tags[cpt] = tag.replace("B", "I")
        if tag.startswith("B") and not temp.startswith("O") and tag[1:] == temp[1:]:
            tags[cpt] = tag.replace("B", "I")
        temp = tags[cpt]
        cpt += 1

    # print("tags after =", tags)

    pos_tags = [pos for token, pos in pos_tag(encode)]
    conlltags = [(token, pos, tg) for token, pos, tg in zip(encode, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)
    annotated_text = []

    for subtree in ne_tree:
        # skipping 'O' tags
        if type(subtree) == Tree:
            # print('subtree:', subtree)
            # print('subtree leaves:', subtree.leaves())
            original_label = subtree.label()
            # print("orignial_label =", original_label)
            original_text = " ".join([token for token, pos in subtree.leaves()])
            # print("original_text =", original_text)
            start_words = [word for word in original_text.split(" ") if "▁" in word]
            # print("start_words =", start_words)
            entities = []

            if original_text != "▁" and original_text.startswith("▁"):
                entities = [entity.replace(" ", "") for entity in original_text.split("▁") if len(entity) != 0]
                original_text = " ".join([entity for entity in entities])
                annotated_text.append(
                    Annotation(original_text, original_label)
                )
            # print("-" * 50)

    finalTokens = []
    tokens = []
    if len(annotated_text) != 0:
        for annotation in annotated_text:
            # print(annotation.entity)
            # print('sentence:', sentence)
            tokens = sentence.split(annotation.entity, 1)
            # print('len(tokens)', len(tokens), 'tokens', tokens)
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
