import time
import jax
import jax.numpy as jnp
import optax
from jax import value_and_grad
import numpy as np


## load and preprocess the dataset
class Processor():

    def __init__(
            self,
            path: str = 'input.txt'
        ):

        with open(path,'r',encoding='utf-8') as f:
            self.data = f.read()

        # print('-- Dataset preview first 100 characters--')
        # print()
        # print(self.data[:100])

        ## get a list of unique characters in the dataset for forming character level token
        chars = sorted(list(set(self.data)))
        print(f'vocab size {len(chars)}')

        ## map characters to integrs
        self.char_to_int = {v:idx for idx,v in enumerate(chars)}
        ## map int to chars
        self.int_to_char = {idx:v for idx,v in enumerate(chars)}

    ## encode the text string into 
    def encode(
            self,
            text:str
    ):
        
        return [self.char_to_int[s] for s in text]
    
    def decode(
            self,
            tokens : list
    ):
        return ''.join([self.int_to_char[t] for t in tokens])

    
    def get_batch(
        self,
        prng : jax.random.key = jax.random.key(42),
        batch_size : int = 4,
        block_size : int = 8
        ):

        ## encode the dataset into integers
        encoded_data = jnp.array(self.encode(self.data), dtype= jnp.int16)
        ## get train/val split
        train_split = int(len(encoded_data) * 0.9)
        train_data = encoded_data[:train_split]
        val_data = encoded_data[train_split:]

        ## get the starting indices of random batches
        train_idx = jax.random.randint(prng,(batch_size,),0,len(train_data)-block_size)
        val_idx = jax.random.randint(prng,(batch_size,),0,len(val_data)-block_size)

        
        x_train = jnp.vstack([train_data[idx:idx+block_size] for idx in train_idx])
        y_train = jnp.vstack([train_data[idx+1:idx+block_size+1] for idx in train_idx])
        x_val = jnp.vstack([val_data[idx:idx+block_size] for idx in val_idx])
        y_val = jnp.vstack([val_data[idx+1:idx+block_size+1] for idx in val_idx])

        return x_train,y_train,x_val,y_val

    def generate_text(
        self,
        model : jnp.array, 
        seq_len : int =100
        ):

        key = jax.random.key(int(time.time()))
        key,key2 = jax.random.split(key)
        sos_token = jax.random.randint(key=key2,shape=(1,),minval=0,maxval=len(self.char_to_int)).item()
        text_tokens = [sos_token]
        next_token = sos_token
        for i in range(seq_len):
            # get the raw logits prediction
            # apply softmax to convert into probabilty distribution and get the next token pred sampled from the distribution
            key,key3 = jax.random.split(key)
            prob = jax.nn.softmax(model[next_token],axis=0)
            next_token = jax.random.choice(key=key3,a=len(prob),p=prob).item()
            text_tokens.append(next_token)

        return self.decode(text_tokens)
    
    
def embedding_model(
        key : jax.random,
        vocab_size : int = 65
        ) :
    
    model = jax.random.normal(key,shape=(vocab_size,vocab_size))
    return model

# ref used for implementing the embedding forwards pass :
# https://stackoverflow.com/questions/72817730/what-is-the-recommended-way-to-do-embeddings-in-jax
def forward(embed_model,
            x : jnp.array, # single x of shape (batch_size,block_size)
            ) -> jnp.array :
    
    # vmap to support batch operation in x_train 
    # ref : https://dinocausevic.com/2023/06/13/jax-vmap/
    # in_axes as  None means that axes will not be iterated over will stay fixed during the iteration process
    # add detailed comment

    output = jax.vmap(lambda embed_model,x: embed_model[x], in_axes=[None,0],out_axes=0)(embed_model,x)
    batch, block, vocab = output.shape # shape (batch_size,block_size,vocab_size)
    output = output.reshape(batch*block, vocab) # shape (batch_size*block_size,vocab_size)

    return output

def compute_loss(
        embed_model : jnp.array, # shape(vocab_size,vocab_size)
        x_train : jnp.array, # single x of shape (batch_size,block_size)
        targets : jnp.array # shape: (batch_size*block_size)
) :
    logits = forward(embed_model,x_train)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits,targets).mean()
    
    return loss

@jax.jit
def train_step(
    model : jnp.array, # embedding model 
    x_train : jnp.array,  # shape (batch,block)
    y_train : jnp.array, # shape (batch*block)
    lr : float = 1e-1,
    ):

    # use jax value_and_grad to compute the gradients
    loss, grad = value_and_grad(compute_loss)(model,x_train,y_train)
    #loss = compute_loss(logits,y_train)
    # update the model parameters
    model = model - lr*grad
    # return the loss and the updated embedding model array
    return loss,model

def train(steps=100):

    model = embedding_model(jax.random.key(0))
    processor = Processor()
    print('Model Output with seqeunce length of 100 after initialization')
    print(processor.generate_text(model,seq_len=1000))
    for step in range(steps): 
        x_train,y_train,x_val,y_val = processor.get_batch()
        targets = y_train.reshape(y_train.shape[0]*y_train.shape[1])
        loss,model = train_step(model,x_train,targets)
        print(f'step: {step} train loss: {loss}')
        #print(f'grad shape: {grad.shape}')
    print(f'Model Output with seqeunce length of 100 after training for {steps} steps:')  
    print(processor.generate_text(model,seq_len=1000))
    #return model


    

if __name__ == '__main__' :

    # processor = Processor()
    # x_train,y_train,x_val,y_val = processor.get_batch()

    # print(f'train input shape: {x_train.shape}, target shape: {y_train.shape}')
    # print(f'val input shape: {x_val.shape}, target shape: {y_val.shape}')
    # print('input elements:')
    # print('train data:',x_train)
    # print('train targets:',y_train)

    # model = embedding_model(jax.random.key(0))
    # print(f'embedding model shape: {model.shape}')
    
    train(1000)


    # print(x_train[0])
    # a = jnp.take(model,x_train[0],axis=0)
    # #print(embedding_model(x_train[0]))
    # b = jax.vmap(lambda model,x: model[x], in_axes=[None,0],out_axes=0)(model,x_train)
    # print(a.shape, b.shape)
    #print(jnp.array_equal(a,b))















        







