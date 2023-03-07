import torch
import csv
import re
import torch.nn.functional as F
from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    
    def __init__(
                self, data_path="Data/eng_ita.tsv",
                start_token='<BoS>', end_token='<EoS>'
                ):
        super(LanguageDataset, self).__init__()
        self.path = data_path
        self.start_token = start_token
        self.end_token = end_token
        self._load_dataset()
        self._tokenize()

    def __getitem__(self, i):
        x = F.one_hot(torch.tensor(self.eng_tokenized[i]), self.eng_voc_size)
        y = F.one_hot(torch.tensor(self.ita_tokenized[i]), self.ita_voc_size)
        return x, y
        
    def __len__(self):
        return len(self.eng)
        
    def _load_dataset(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            f = csv.reader(f, delimiter='\t')
            data = [line for line in f]
        self.eng, self.ita = [l[1] for l in data], [l[3] for l in data]
        
    def _split(self, sentence):
        sentence.lower()
        sentence = re.sub(r'[,.:;\-""!%&?\/]', r' ', sentence)
        sentence = re.sub(r'([0-9]+) *([€$£])', r'\2\1', sentence)
        sentence = sentence.strip()
        sentence = re.split(r'[ \']+', sentence)
        sentence.insert(0, self.start_token)
        sentence.append(self.end_token)
        return sentence
    
    def _tokenize(self):
        eng_tokenized = [self._split(sentence) for sentence in self.eng]
        ita_tokenized = [self._split(sentence) for sentence in self.ita]
        
        eng = set(word for sentence in eng_tokenized for word in sentence)
        eng.add((self.start_token, self.end_token))
        ita = set(word for sentence in ita_tokenized for word in sentence)
        eng.add((self.start_token, self.end_token))
        
        self.eng_voc_size = len(eng)
        self.ita_voc_size = len(ita)
        
        self.to_eng = {idx: word for idx, word in enumerate(eng)}
        self.from_eng = {word: idx for idx, word in enumerate(eng)}
        self.to_ita = {idx: word for idx, word in enumerate(ita)}
        self.from_ita = {word: idx for idx, word in enumerate(ita)}
        
        self.eng_tokenized = [[self.from_eng[word] for word in sentence] for sentence in eng_tokenized]
        self.ita_tokenized = [[self.from_ita[word] for word in sentence] for sentence in ita_tokenized]
