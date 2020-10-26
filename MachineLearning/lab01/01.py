texts = [
    "Life is like Machine Learning",
    "We are learning from experience"]

def word_map(texts):
    words_dic = {}
    for s in texts:
        str_list = s.split()
        for w in str_list:
            w = w.lower()
            if w not in words_dic:
                words_dic[w] = 1
            else:
                words_dic[w] += 1

    return words_dic

result = word_map(texts)
for key in sorted(result.keys()):
    print(key, ":", result[key])