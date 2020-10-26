texts = [
    "Life is like Machine Learning",
    "We are learning from experience"]



def uncommon_word_map(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique uncommon words and their occurrences over texts
    """
    with open("MachineLearning\\lab01\\commonwords.txt", "r") as file1:
        FileasList = file1.readlines()
    for i in range(len(FileasList)):
        FileasList[i] = FileasList[i][:-1]
        
    words_dic = {}
    for s in texts:
        str_list = s.split()
        for w in str_list:
            w = w.lower()
            if w in FileasList:
                continue
            if w not in words_dic:
                words_dic[w] = 1
            else:
                words_dic[w] += 1

    return words_dic

result = uncommon_word_map(texts)
for key in sorted(result.keys()):
    print(key, ":", result[key])