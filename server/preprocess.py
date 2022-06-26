# Format data
def format_text(text):
    lines = text.split("\n")
    data = []
    for line in lines:
        for token in line.split(" "):
            data.append(token)
    return data


# Feature set
def token2features(tokens, i):
    word = tokens[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = tokens[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(tokens)-1:
        word1 = tokens[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


# Get features set by sentence
def text2features(sent):
    return [token2features(sent, i) for i in range(len(sent))]
