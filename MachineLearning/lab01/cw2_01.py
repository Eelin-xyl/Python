def word_map(texts):
    """
    Inputs a list of string texts
    Returns a dictionary of unique words and their occurrences over texts
    """
    # Write your code here
    
    
    


def uncommon_word_map(texts):
    """
    Inputs a list of string texts
    Returns a dictionary of unique uncommon words and their occurrences over texts
    """
    # Write your code here
    
    
    


def check_word_map():
    texts = [
        "Life is like Machine Learning",
        "We are learning from experience"]
    result = word_map(texts)
    for key in sorted(result.keys()):
        print(key, ":", result[key])


def check_uncommon_word_map():
    texts = [
        "Life is like Machine Learning",
        "We are learning from experience"]
    result = uncommon_word_map(texts)
    for key in sorted(result.keys()):
        print(key, ":", result[key])


def main():
    check_word_map()
    #check_uncommon_word_map()


if __name__ == "__main__":
    main()
