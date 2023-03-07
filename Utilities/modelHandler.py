import torch

class LSTModel():
    
    def __init__(self, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, eos_token, bos_token):
        
        self.encoder = encoder
        self.decoder = decoder
        self.enc_opt = encoder_optimizer
        self.dec_opt = decoder_optimizer
        self.loss_fun = loss_function
        self.eos_token = eos_token
        self.bos_token = bos_token
        
    def train_model(self, dataloader, max_epochs, save_every_epochs, ckp_name, embedding_size):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialise models
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.train()
        self.decoder.train()
                
        total_training_loss = []
        
        hidden_encoder = torch.zeros(1,1,embedding_size, device = device) # Embedding size
        
        for e in range(max_epochs):
            
            total_loss_epoch = 0
            
            for id_batch, batch in enumerate(dataloader):
                
                self.enc_opt.zero_grad()
                self.dec_opt.zero_grad()
                
                idx, X, y = batch
                # Dimesions: X - 1 x seq_input_length x embed_size
                #            y - 1 x seq_output_length x embed_size
                X, y = X.to(device), y.to(device)
                
                seq_length = X.size()[1]
                
                # ENCODER
                output_encoder, hidden_encoder = self.encoder(X, hidden_encoder)              
                output_encoder = output_encoder[:, -1, :].unsqueeze(1) # take the last hidden state
                
                # DECODER
                
                input_decoder = torch.tensor([[self.bos_token]], device = device)
                
                output_decoder, hidden_decoder = decoder(input_decoder, output_encoder) 
                
                
                if decoder_output == self.eos_token:
                    
                    continue
                
                
                total_loss_epoch += loss
                
            total_loss_epoch /= seq_length
            
            total_training_loss.append(total_loss_epoch.item())
                
            
            
            ## SAVE A CHECKPOINT
            if e%save_every_epochs == 0: # save the model every "save_every_epochs" epochs
                ckp_path_enc = ckp_name+'_enc_{}.pth'.format(e)
                ckp_path_dec = ckp_name+'_dec_{}.pth'.format(e)
                torch.save(self.encoder.state_dict(), ckp_path_enc)
                torch.save(self.decoder.state_dict(), ckp_path_dec)  
                
                
        return total_training_loss
        
    
        
        

    