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


async def predict_ner_bioclinicalbert(sentence):
    model_id = "data/clinical_ner_Bio_ClinicalBert"
    # model_id = "data/clinical_ner_bluebert_pubmed_mimic_uncased"
    # model_id = "raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed"

    sentence = sentence.lower()
    # print(sentence)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    id2label = {0: 'O', 1: 'B-disease', 2: 'I-disease'}

    """
    sentence = "un homme âgé de 61 ans."
    print(sentence[0:5], len(sentence[0:5]))
    print(sentence[6:9], len(sentence[6:9]))
    print(sentence[10:12], len(sentence[10:12]))
    print(sentence[13:15], len(sentence[13:15]))
    print(sentence[16:20], len(sentence[16:20]))
    """

    inputs = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")

    encode = tokenizer.tokenize(sentence)
    decode = tokenizer.decode(inputs['input_ids'][0])

    # print(decode)

    flat_pred = []

    with torch.no_grad():
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        flat_pred.extend(np.argmax(logits, axis=-1).flatten())

    predictions = [id2label[id] for id in flat_pred if id != -100]
    '''
    print(len(flat_pred), len(predictions), len(inputs['offset_mapping'][0]))
    print('Sentence: ', decode)
    print('Labels', flat_pred)
    print('Predictions', predictions)
    print('Encode', encode)
    for code, prediction, map in zip(encode, predictions[1:len(predictions)-1], inputs['offset_mapping'][0][1:23]):
        print(code, prediction, map)
    # print('sentence:', len(encode), 'predictions:', len(predictions), 'sentence:', len(sentence.split(" ")))
    '''
    # print('encode:', encode)

    tags = predictions[1:len(predictions) - 1]
    '''
    for code, tag in zip(encode, tags):
        if tag != 'O':
            print("code: ", code, "tag: ", tag)
    '''

    pos_tags = [pos for token, pos in pos_tag(encode)]
    # print(len(pos_tags))
    # print('pos_tags:', pos_tags)
    conlltags = [(token, pos, tg) for token, pos, tg in zip(encode, pos_tags, tags)]
    # print('conlltags:', conlltags)
    ne_tree = conlltags2tree(conlltags)
    # print('ne_tree:', ne_tree)
    annotated_text = []
    for subtree in ne_tree:
        # skipping 'O' tags
        if type(subtree) == Tree:
            # print('subtree:', subtree)
            # print('subtree leaves:', subtree.leaves())
            original_label = subtree.label()
            original_text = " ".join([token for token, pos in subtree.leaves()])
            # print(f'subtree: {subtree}, original_text: {original_text}')
            start_words = [word for word in original_text.split(" ") if "#" not in word]
            # print("original text:", original_text)
            # print(start_words)
            entities = []
            if len(start_words) <= 2:
                if len(start_words) == 2:
                    start = original_text.index(start_words[0])
                    end = original_text.index(start_words[1])
                    entities.append(original_text[start:end].replace(" ##", ""))
                    entities.append(original_text[end:].replace(" ##", ""))
                else:
                    entities = [entity for entity in original_text.split("##") if len(entity) != 0]
                # print("entities: ", entities)
                # print("-----------------------------------------------------------------------------------------------")
                # original_text = "".join([entity for entity in entities])
                annotated_text.append(
                    Annotation(original_text, original_label)
                )

    finalTokens = []
    tokens = []
    '''
    for entity in annotated_text:
        print(entity.entity, entity.tag)
    '''

    # print(annotated_text)
    if len(annotated_text) != 0:
        # tokensList = sentence
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
                    # print("annotation:", annotation, " index:", annotated_text.index(annotation))
                    if annotated_text.index(annotation) == len(annotated_text) - 1:
                        sentence = ""

        if len(sentence) != 0:
            finalTokens.append(Annotation(sentence))

    # print("sentence: ", sentence)
    '''
    for token in finalTokens:
        print(token.entity, token.tag)
    '''
    return finalTokens
