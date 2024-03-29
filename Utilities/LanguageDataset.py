import torch
import csv
import re
import random
import torch.nn.functional as F
from torch.utils.data import Dataset

# Class to handle the language dataset, perfoming all the preprocessing operations
# and crating the train, validation and test datasets
class LanguageDataset(Dataset):
    
    def __init__(
                self, data_path="Data/eng_ita.tsv",
                start_token='<BoS>', 
                end_token='<EoS>',
                pad_token='<UNK>',
                seq_len=20,
                test_split = 0.2,
                val_split = 0.1,
                shuffle_data = True,
                limit_data = None
                ):
        super(LanguageDataset, self).__init__()
        self.path = data_path
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.seq_len = seq_len
        self.test_split = test_split 
        self.val_split = val_split 
        self.shuffle_data = shuffle_data
        self.limit_data = limit_data
           
    def get_datasets(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            f = csv.reader(f, delimiter='\t')
            data = []
            
            # To limit the lines considered when loading the dataset
            if self.limit_data is None:
                self.limit_data = 1e9
            
            for idd, line in enumerate(f):
                if idd < self.limit_data:
                    data.append(line)
                else:
                    break
                     
        self.corpus_dict = {}
        eng = []
        ita = []
        for l in data:
            
            eng.append(l[1])
            ita.append(l[3])
            
            if l[1] not in self.corpus_dict.keys():
                self.corpus_dict[l[1]] = [l[3]]
            else:
                self.corpus_dict[l[1]].append(l[3])

        # Create vocabularies with all words for both language
        self._create_vocabulary(eng, ita)
        
        # Split
        keys = list(self.corpus_dict.keys())
        if self.shuffle_data is True:
            random.shuffle(keys)

        # Split the keys in train, validation and test from english keys
        keys_test = keys[0 : int(len(keys)*self.test_split)]
        keys_val = keys[len(keys_test) : len(keys_test)+int(len(keys)*self.val_split)]
        keys_train = keys[len(keys_test) + len(keys_val) : len(keys)]
        
        # Dictionaries for BLUE scores
        self.blue_score_test = {' '.join(self._split_key(key)): [self._split_key(c) for c in self.corpus_dict[key]] for key in keys_test}      
        
        # Divide data according to split in keys
        eng_train, ita_train = self._dataset_split(keys_train)
        eng_val, ita_val = self._dataset_split(keys_val)
        eng_test, ita_test = self._dataset_split(keys_test)
        
        # Tokenize and create the datasets
        train_set = self._create_tokenized_set(eng_train, ita_train)
        val_set = self._create_tokenized_set(eng_val, ita_val)
        test_set = self._create_tokenized_set(eng_test, ita_test)
    
        return train_set, val_set, test_set
        
    def _pad_sequence(self, sequence):
        if len(sequence) >= self.seq_len - 2:
            sequence = sequence[ : self.seq_len - 2]
            sequence.insert(0, self.start_token)
            sequence.append(self.end_token)
        else:
            sequence.insert(0, self.start_token)
            sequence.append(self.end_token)
            sequence += [self.pad_token for _ in range(self.seq_len - len(sequence))]
        return sequence
    
    def _split_key(self, sentence):
        
        # Split the sentences used for the BLEU score metric
        sentence.lower()
        sentence = re.sub(r'[,.:;\-""!%&?\/]', r' ', sentence)
        sentence = re.sub(r'([0-9]+) *([€$£])', r'\2\1', sentence)
        sentence = sentence.strip()
        sentence = re.split(r'[ \']+', sentence)
        
        if len(sentence) >= (self.seq_len - 2) :
            sentence = sentence[ : self.seq_len - 2]

        return sentence
    
    def _split(self, sentence):
        
        # Split the sentences used for training, validation and testing
        sentence.lower()
        sentence = re.sub(r'[,.:;\-""!%&?\/]', r' ', sentence)
        sentence = re.sub(r'([0-9]+) *([€$£])', r'\2\1', sentence)
        sentence = sentence.strip()
        sentence = re.split(r'[ \']+', sentence)
        sentence = self._pad_sequence(sentence)
        
        return sentence
    
    def _create_tokenized_set(self, eng, ita):
        
        # Tokenike English and Italian sentences
        eng_tokenized = [self._split(sentence) for sentence in eng]
        ita_tokenized = [self._split(sentence) for sentence in ita]
        
        eng_tokenized = [[self.from_eng[word] for word in sentence] for sentence in eng_tokenized]
        ita_tokenized = [[self.from_ita[word] for word in sentence] for sentence in ita_tokenized]
             
        data_set = SplitDataset(eng_sentences = eng_tokenized,
                                ita_sentences = ita_tokenized
                               )
        
        return data_set
    
    def _dataset_split(self, keys):
        
        # Split the corpus
        eng = []
        ita = []
        corpus_dict_temp = self.corpus_dict.copy()

        for key in keys:
            pair = corpus_dict_temp[key].copy()
            while len(pair) != 0:
                eng.append(key)
                ita.append(pair.pop())
                
        return eng, ita
    
    def _create_vocabulary(self, eng, ita):
        
        # Create English and Italian vocabularies 
        eng_tokenized = [self._split(sentence) for sentence in eng]
        ita_tokenized = [self._split(sentence) for sentence in ita]
        
        eng = set(word for sentence in eng_tokenized for word in sentence)
        eng.add((self.start_token, self.end_token, self.pad_token))
        ita = set(word for sentence in ita_tokenized for word in sentence)
        eng.add((self.start_token, self.end_token, self.pad_token))
        
        self.eng_voc_size = len(eng)
        self.ita_voc_size = len(ita)
        self.to_eng = {idx: word for idx, word in enumerate(eng)}
        self.from_eng = {word: idx for idx, word in enumerate(eng)}
        self.to_ita = {idx: word for idx, word in enumerate(ita)}
        self.from_ita = {word: idx for idx, word in enumerate(ita)}
        
    def translate(self, X, language):
        
        # Create the translations in English or Italian from the tokens 
        if language == 'ita':
            lang = self.to_ita
        elif language == 'eng':
            lang = self.to_eng
        else:
            print("Select ita or eng")
            return
        
        X_out = []
        
        for line in X:
            
            text = []
            for letter in line:

                if lang[letter.item()] == self.start_token or lang[letter.item()] == self.pad_token:
                    pass
                elif lang[letter.item()] == self.end_token:
                    break
                else:
                    text.append(lang[letter.item()])
            
            X_out.append(text)

        return X_out
               
class SplitDataset(Dataset):
    
    # Class to handle the split datasets
    def __init__(
            self, 
            eng_sentences = None,
            ita_sentences = None
            ):
        super(SplitDataset, self).__init__()
        self.eng_sentences = eng_sentences
        self.ita_sentences = ita_sentences
        
    def __getitem__(self, i):
        x = torch.tensor(self.eng_sentences[i])
        y = torch.tensor(self.ita_sentences[i])
        return x, y
        
    def __len__(self):
        return len(self.eng_sentences)