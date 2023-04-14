import torch
import csv
import re
import torch.nn.functional as F
from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    
    def __init__(
                self, data_path="Data/eng_ita.tsv",
                start_token='<BoS>', end_token='<EoS>', pad_token='<UNK>',
                seq_len=20
                ):
        super(LanguageDataset, self).__init__()
        self.path = data_path
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.seq_len = seq_len
        
        self._load_dataset()
        self._tokenize()

    def __getitem__(self, i):
        x = torch.tensor(self.eng_tokenized[i])
        y = torch.tensor(self.ita_tokenized[i])
        return x, y
        
    def __len__(self):
        return len(self.eng)
        
    def _load_dataset(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            f = csv.reader(f, delimiter='\t')
            data = []
            for idd, line in enumerate(f):
                if idd <500:
                    data.append(line)
            #data = [line for line in f]
        self.eng, self.ita = [l[1] for l in data], [l[3] for l in data]
        
    def _pad_sequence(self, sequence):
        if len(sequence) >= self.seq_len - 1:
            sequence = sequence[ : self.seq_len - 1]
        else:
            sequence += [self.pad_token for _ in range(self.seq_len - len(sequence) - 1)]
        return sequence
    
    def _split(self, sentence):
        sentence.lower()
        sentence = re.sub(r'[,.:;\-""!%&?\/]', r' ', sentence)
        sentence = re.sub(r'([0-9]+) *([€$£])', r'\2\1', sentence)
        sentence = sentence.strip()
        sentence = re.split(r'[ \']+', sentence)

        sentence.insert(0, self.start_token)
        sentence = self._pad_sequence(sentence)
        sentence.append(self.end_token)
        return sentence
    
    def _tokenize(self):
        eng_tokenized = [self._split(sentence) for sentence in self.eng]
        ita_tokenized = [self._split(sentence) for sentence in self.ita]
        
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
        
        self.eng_tokenized = [[self.from_eng[word] for word in sentence] for sentence in eng_tokenized]
        self.ita_tokenized = [[self.from_ita[word] for word in sentence] for sentence in ita_tokenized]

    
    def translate(self, token, language):
        
        if language == 'ita':
            lang = self.to_ita
        elif language == 'eng':
            lang = self.to_eng
        else:
            print("Select ita or eng")
            return
            
        string = ""
        for letter in token:
            string += " "
            string += lang[letter.item()]
        return string
    
def my_collate_fn(batch):
    # x = [item[0] for item in batch]
    # target = [item[1] for item in batch]
    # return [x, target]
    return batch
