from collections import defaultdict

LINES = 1000

def make_bitext(num_lines):
    french_sentences = []
    english_sentences = []
    with open("data/hansards.f") as f:
        i = 0
        for line in f:
            french_sentences.append(line.strip().split())
            if i > num_lines:
                break
            i += 1

    with open("data/hansards.e") as f:
        i = 0
        for line in f:
            english_sentences.append(line.strip().split())
            if i > num_lines:
                break
            i += 1
    bitext = [[french_sentences[i], english_sentences[i]] for i in range(num_lines)]
    return bitext

class IBModel1:
    def __init__(self, english_sentences, french_sentences):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        
        self.all_english_words = set()
        for sentence in english_sentences:
            for word in sentence:
                self.all_english_words.add(word)

        self.all_french_words = set()
        for sentence in french_sentences:
            for word in sentence:
                self.all_french_words.add(word)

        # initialize t values
        self.t = {}  
        for e_word in self.all_english_words: 
            self.t[e_word] = {}
            for f_word in self.all_french_words:
                self.t[e_word][f_word] = 1/len(self.all_french_words)

    def EM(self, bitext):
        count = {}
        total = {}
        for e_word in self.all_english_words:
            count[e_word] = {}
            for f_word in self.all_french_words:
                count[e_word][f_word] = 0

        for f_word in self.all_french_words:
            total[f_word] = 0 

        for sentence_pair in bitext:
            french_sentence = sentence_pair[0]
            english_sentence = sentence_pair[1]

            # compute normalization
            s_total = {}
            for e_word in english_sentence:
                s_total[e_word] = 0
                for f_word in french_sentence:
                    s_total[e_word] += self.t[e_word][f_word]

            # collect counts 
            for e_word in english_sentence: 
                for f_word in french_sentence:
                    count[e_word][f_word] += self.t[e_word][f_word] / s_total[e_word]
                    total[f_word] += self.t[e_word][f_word] / s_total[e_word]

        # estimate probabilities
        for f_word in self.all_french_words:
            for e_word in self.all_english_words:
                self.t[e_word][f_word] = count[e_word][f_word] / total[f_word]


# if __name__ == "__main__":
    

