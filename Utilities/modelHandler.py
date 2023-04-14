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
        
        for e in range(1,max_epochs+1):
            
            mean_loss_epoch = 0
            
            for id_batch, batch in enumerate(dataloader): 

                self.enc_opt.zero_grad()
                self.dec_opt.zero_grad()
                
                loss_batch = 0 
                
                # Dimesions: X : batch_size x input_seq_length 
                #            y : batch_size x output_seq_length
                X, y = batch 
                X, y = X.to(device), y.to(device)
                #print("Input enc : {}".format( X.size()))
                
                # Sequence length 
                input_seq_length = X.size()[1]
                output_seq_length = y.size()[1]
                batch_size = X.size()[0]
                
                ### ENCODER ###
                output_encoder, hidden_encoder = self.encoder(X) # output: batch_size x input_seq_length x emb_size
                last_hidden_encoder = output_encoder[:, -1, :].unsqueeze(0) # take the last hidden state: 1 x batch_size x emb_size
                c_final = hidden_encoder[1] # Encoder final state 
                
                #print("Last hidden state : {}".format(last_hidden_encoder.size()))
                
                ### DECODER ###
                # Give token <BoS> as input in the decoder
                input_decoder = torch.tensor([[self.bos_token for _ in range(batch_size)]], device = device) # dimension: 1 x batch_size
                input_decoder =  torch.transpose(input_decoder, 0, 1) # dimension: batch_size x 1
                
                #print("Input dec : {}".format( input_decoder.size()))
                
                output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_encoder) # ouput: batch_size x vocabulary
                #print("Out dec : {}".format( output_decoder.size()))
                
                prediction = output_decoder.topk(1).indices
                #print("prediction : {}".format( prediction.size()))
                
                target = y[:, 1]
                
                loss = self.loss_fun(output_decoder.squeeze(1), target)
                loss_batch += loss
                    
                for idt in range(2, output_seq_length):

                    output_decoder, hidden_decoder = self.decoder(prediction.squeeze(1), hidden_decoder)
                    prediction = output_decoder.topk(1).indices
                    target = y[:, idt]

                    loss_batch += self.loss_fun(output_decoder.squeeze(1), target)

                loss_batch /= output_seq_length
                    
               # End of batch 
                                    
                loss_batch.backward()
                
                self.enc_opt.step()
                self.dec_opt.step()
                
                mean_loss_epoch += loss_batch / len(dataloader)
            
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
        
        device = torch.device("cpu")# if torch.cuda.is_available() else "cpu")
        
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
                
                X, y = batch 
                X, y = X.to(device), y.to(device)
                batch_size = X.size()[0]
                seq_length = X.size()[1]
                trans = []
                
                ### ENCODER ###
                output_encoder, hidden_encoder = self.encoder(X) # output: batch_size x input_seq_length x emb_size
                
                ### DECODER ###
                # Give token <BoS> as input in the decoder
                input_decoder = torch.tensor([[self.bos_token for _ in range(batch_size)]], device = device) # dimension: 1 x batch_size
                input_decoder =  torch.transpose(input_decoder, 0, 1) # dimension: batch_size x 1
                trans.append(input_decoder)

                output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_encoder) # ouput: vocab_size

                output_decoder = output_decoder.softmax(dim=-1)
                
                prediction = output_decoder.topk(1).indices
                
                trans.append(prediction.squeeze(-1))

                for _ in range( seq_length - 2):

                    output_decoder, hidden_decoder = self.decoder(prediction.squeeze(1), hidden_decoder)
                    output_decoder = output_decoder.softmax(dim=-1)
                    prediction = output_decoder.topk(1).indices
                    trans.append(prediction.squeeze(-1))
                    
                trans = torch.cat(trans, dim=-1)
                return X, y, trans
                
                trans = torch.cat(trans, dim=-1)

                # For each row, return the original sentences and the translation
                result.append((X,y,trans))

        return result











    