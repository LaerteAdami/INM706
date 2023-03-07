from torch.utils import data

class FakeNewsDataset(data.Dataset):
    
    def __init__(self, directory = None,
                 stage  = None,
                 stop_words = None,  
                 start_token  = None, 
                 end_token  = None):
        
        self.dir = directory 
        self.stage = stage
        self.stop_words = stopwords.words('english')
        self.start_token = start_token
        self.end_token = end_token
        
        
    def load_tweet(self, idx):
        
        pass
        
        
        
    
        
    def __len__(self):
        
        return
    
    def __get_item__(self, idx):
        
        return
