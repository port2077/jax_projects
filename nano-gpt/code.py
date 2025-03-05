import jax
import jax.numpy as jnp
import numpy as np


## load and preprocess the dataset
class Processor():

    def __init__(
            self,
            path: str = 'input.txt'
        ):

        with open(path,'r',encoding='utf-8') as f:
            self.data = f.read()

        print('-- Dataset preview first 100 characters--')
        print()
        print(self.data[:100])

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
    

if __name__ == '__main__' :

    processor = Processor()
    x_train,y_train,x_val,y_val = processor.get_batch()

    print(f'train input shape: {x_train.shape}, target shape: {y_train.shape}')
    print(f'val input shape: {x_val.shape}, target shape: {y_val.shape}')
    print('input elements:')
    print('train data:',x_train)
    print('train targets:',y_train)











        







