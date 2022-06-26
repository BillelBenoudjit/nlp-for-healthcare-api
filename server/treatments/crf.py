import pickle

# import nltk
# nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree


class Annotation:
    def __init__(self, entity, tag=""):
        self.entity = entity
        self.tag = tag


def crf(text, tokens, features, model_id):
    model = pickle.load(open(model_id, 'rb'))
    predictions = model.predict(features)

    pos_tags = [pos for token, pos in pos_tag(tokens)]
    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, predictions[0])]
    ne_tree = conlltags2tree(conlltags)
    annotated_text = []
    for subtree in ne_tree:
        # skipping 'O' tags
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_text = " ".join([token for token, pos in subtree.leaves()])
            annotated_text.append(
                Annotation(original_text, original_label)
            )

    final_annotations = []
    if len(annotated_text) != 0:
        for annotation in annotated_text:
            tokens = text.split(annotation.entity, 1)
            if len(tokens) == 2:
                final_annotations.append(Annotation(tokens[0]))
                final_annotations.append(annotation)
                if len(tokens[1]) != 0:
                    text = tokens[1]
                else:
                    if annotated_text.index(annotation) == len(annotated_text) - 1:
                        text = ""
        if len(text) != 0:
            final_annotations.append(Annotation(text))
    return final_annotations
