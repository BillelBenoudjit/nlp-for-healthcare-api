from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import torch

async def predict(sentence):
    # sentence = "un homme âgé de 61 ans."

    tokenizer = AutoTokenizer.from_pretrained("data/clinical_ner_camembert_40")

    inputs = tokenizer(sentence, return_tensors="pt")

    print(inputs)

    decode = tokenizer.decode(inputs['input_ids'][0])
    print(decode)

    flat_pred = []
    id2label = {0: 'B-age', 1: 'B-anatomie', 2: 'B-frequence', 3: 'I-valeur', 4: 'B-sosy', 5: 'I-mode', 6: 'I-duree',
     7: 'B-moment', 8: 'B-origine', 9: 'I-frequence', 10: 'I-traitement', 11: 'I-origine', 12: 'O', 13: 'I-age',
     14: 'B-examen', 15: 'I-issue', 16: 'B-pathologie', 17: 'I-genre', 18: 'I-moment', 19: 'I-sosy', 20: 'B-date',
     21: 'I-pathologie', 22: 'B-mode', 23: 'I-anatomie', 24: 'I-examen', 25: 'B-issue', 26: 'I-substance', 27: 'B-dose',
     28: 'I-date', 29: 'I-dose', 30: 'B-valeur', 31: 'B-traitement', 32: 'B-duree', 33: 'B-substance', 34: 'B-genre'}

    with torch.no_grad():
        model = AutoModelForTokenClassification.from_pretrained("data/clinical_ner_camembert_40")
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        flat_pred.extend(np.argmax(logits, axis=-1).flatten())

    predictions = [id2label[id] for id in flat_pred if id != -100]

    print('Sentence: ', decode)
    print('Labels', flat_pred)
    print('Predictions', predictions)
    return predictions
