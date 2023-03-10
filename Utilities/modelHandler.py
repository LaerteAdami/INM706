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
        
    def train_model(self, dataloader, max_epochs, save_every_epochs, ckp_name):
        
        device = torch.device("cpu")# if torch.cuda.is_available() else "cpu")
        
        # Initialise models
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.train()
        self.decoder.train()
                
        total_training_loss = []
        
        #hidden_encoder = torch.zeros(1,1,embedding_size, device = device) # Embedding size
        #ht_0 = torch.zeros(1,embedding_size, device = device)
        #ct_0 = torch.zeros(1,embedding_size, device = device)
        #hidden_encoder = (ht_0, ct_0)
        
        for e in range(1,max_epochs+1):
            
            mean_loss_epoch = 0
            
            for id_batch, batch in enumerate(dataloader): # take each single word
                
                self.enc_opt.zero_grad()
                self.dec_opt.zero_grad()
                
                total_loss_batch = 0 

                for line in batch:
                    X = line[0]
                    y = line[1]
                    
                    loss_seq = 0
                    
                    # Dimesions: X : seq_input_length 
                    #            y : seq_output_length
                    X, y = X.to(device), y.to(device)

                    input_seq_length = X.size()[0]
                    output_seq_length = y.size()[0]
                    #print(X, y)
                    
                    ### ENCODER ###
                    output_encoder, hidden_encoder = self.encoder(X) # output: seq_input_length x emb_size  
                    #output_encoder = output_encoder[-1, :].unsqueeze(0) # take the last hidden state: 1 x emb_size

                    ### DECODER ###
                    # Give token <BoS> as input in the decoder
                    input_decoder = torch.tensor([self.bos_token], device = device) # dimension: 1
                          
                    output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_encoder) # ouput: vocab_size
                          
                    prediction = output_decoder.topk(1).indices
                    target = y[1] # e.g., 45

                    loss = self.loss_fun(output_decoder, target.unsqueeze(0))
                    loss_seq += loss
                    
                    for idt in range(2, output_seq_length):
                        
                        output_decoder, hidden_decoder = self.decoder(prediction.squeeze(0), hidden_decoder)
                        prediction = output_decoder.topk(1).indices
                        target = y[idt]
                        
                        #print(output_decoder.size(), target.size())

                        loss = self.loss_fun(output_decoder.squeeze(0), target)
                        loss_seq += loss               
                        
                        if prediction == self.eos_token: # EOS index
                            break
                    
                    total_loss_batch += (loss_seq /  output_seq_length ) 
                    
               # End of batch 
                                    
                total_loss_batch.backward()
                
                self.enc_opt.step()
                self.dec_opt.step()
                
                mean_loss_epoch += total_loss_batch / len(dataloader)
            
            total_training_loss.append(mean_loss_epoch.item())
            
            print("Completed epoch: {}, loss: {}".format(e, round(mean_loss_epoch.item(),3)))
                
            ## SAVE A CHECKPOINT
            if e%save_every_epochs == 0: # save the model every "save_every_epochs" epochs
                ckp_path_enc = ckp_name+'_enc_{}.pth'.format(e)
                ckp_path_dec = ckp_name+'_dec_{}.pth'.format(e)
                torch.save(self.encoder.state_dict(), ckp_path_enc)
                torch.save(self.decoder.state_dict(), ckp_path_dec)  
                               
        return total_training_loss
    

    def evaluate_model(self, dataloader, max_length, enc_ckp = None, dec_ckp = None):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if enc_ckp is not None:
            self.encoder.load_state_dict(torch.load(enc_ckp))
        if dec_ckp is not None:
            self.encoder.load_state_dict(torch.load(dec_ckp))
            
        # Initialise models
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()
        
        result = []
                            
        for id_batch, batch in enumerate(dataloader): # take each batch

            for line in batch: # Take each sentence
                
                X = line[0]
                y = line[1]
                trans = []
                
                X, y = X.to(device), y.to(device)

                ### ENCODER ###
                output_encoder, hidden_encoder = self.encoder(X) # output: seq_input_length x emb_size  

                ### DECODER ###
                # Give token <BoS> as input in the decoder
                input_decoder = torch.tensor([self.bos_token], device = device) # dimension: 1
                trans.append(self.bos_token)

                output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_encoder) # ouput: vocab_size

                output_decoder = output_decoder.softmax(dim=1)
                prediction = output_decoder.topk(1).indices
                
                trans.append(prediction)

                for idt in range(max_length):

                    output_decoder, hidden_decoder = self.decoder(prediction.squeeze(0), hidden_decoder)
                    output_decoder = output_decoder.softmax(dim=1)
                    prediction = output_decoder.topk(1).indices
                    trans.append(prediction.item())          

                    if prediction == self.eos_token: # EOS index
                        break

                # For each row, return the original sentences and the translation
                result.append((X,y,torch.tensor(trans)))
            break

        return result











    