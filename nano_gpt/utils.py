import json
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt # type: ignore

import jax
import jax.numpy as jnp
from flax import traverse_util

## helper class that t load train/test data and generate results from trained model
class Processor():

    def __init__(
            self,
            path: str = 'input.txt'
        ):

        with open(path,'r',encoding='utf-8') as f:
            self.data = f.read()

        ## get a list of unique characters in the dataset for forming character level token
        chars = sorted(list(set(self.data)))
        print(f'vocab size {len(chars)}')

        ## map characters to integrs
        self.char_to_int = {v:idx for idx,v in enumerate(chars)}
        ## map int to chars
        self.int_to_char = {idx:v for idx,v in enumerate(chars)}

    ## encode the dataset chars to int
    def encode(
            self,
            text:str
    ):
        
        return [self.char_to_int[s] for s in text]
    
    ## decode from the int to text strings
    def decode(
            self,
            tokens : list
    ):
        return ''.join([self.int_to_char[t] for t in tokens])

    # generate batch of text data
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

        # turns list of slices -> (batch_size, block_size)
        x_train = jnp.vstack([train_data[idx:idx+block_size] for idx in train_idx])
        # same slices as x_train shifted by 1 character -> (batch_size, block_size)
        y_train = jnp.vstack([train_data[idx+1:idx+block_size+1] for idx in train_idx])
        x_val = jnp.vstack([val_data[idx:idx+block_size] for idx in val_idx])
        y_val = jnp.vstack([val_data[idx+1:idx+block_size+1] for idx in val_idx])

        return x_train,y_train,x_val,y_val
    
    # helper function to generate text from the model and decode 
    def generate_text_from_transfomer(
        self,
        params: Dict,
        transformer,
        num_tokens_to_generate: int = 100,
        number_of_layers: int = 4,
    ):
        model_block_size = params['pos_encoding'].shape[0]
        actual_vocab_size = len(self.char_to_int)

        key = jax.random.key(int(time.time()))
        key, sos_key, loop_key = jax.random.split(key, 3)

        sos_token = jax.random.randint(key=sos_key, shape=(1,), minval=0, maxval=actual_vocab_size).item()
        
        current_context_list = [0] * (model_block_size -1) + [sos_token]
        generated_tokens_list = []

        for _ in range(num_tokens_to_generate):
            loop_key, choice_key = jax.random.split(loop_key)
            
            input_sequence = jnp.array(current_context_list[-model_block_size:], dtype=jnp.int32)
            
            
            logits_all_positions = transformer(input_sequence, params, number_of_layers, training=False)
            
            next_token_logits = logits_all_positions[-1] 
            
            probabilities = jax.nn.softmax(next_token_logits, axis=-1)
            next_token = jax.random.choice(key=choice_key, a=actual_vocab_size, p=probabilities).item()
            
            generated_tokens_list.append(next_token)
            
            current_context_list.append(next_token)
            if len(current_context_list) > model_block_size:
                current_context_list.pop(0)

        return self.decode(generated_tokens_list)
    


# this is a helper fx that it used to label the params Pytree with labels which 
# will help to identify which layers needs to be updated by muon or adam
def label_lf(params):
   
    def label_from_path(path_tuple, value):
        # path_tuple will be a tuple of strings/ints, e.g., ('multihead_params_0', 0, 'w_q')
        path_string = "/".join(map(str, path_tuple))
        # only 2D weight matrices are updated by muon 
        # all other veectors and embedding ( which are not dense activations) are updated by adam
        keywords_for_adam = ["gamma", "beta", "bias","embed","positional"]
        
        if any(keyword in path_string for keyword in keywords_for_adam):
            return 'adam'
        else:
            return 'muon'

    return traverse_util.path_aware_map(label_from_path, params)

# helper fx that plots the loss and logs the metrics as a json object
def plot_and_save_loss(steps_list, train_losses, 
                       val_losses, eval_interval, plot_path, log_path):
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps_list, train_losses, label='Training Loss')
        val_steps = steps_list[eval_interval-1::eval_interval]
        plt.plot(val_steps, val_losses, label='Validation Loss')
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(plot_path)
        plt.close()

        loss_data = {
            'steps_list': steps_list,
            'train_losses': train_losses,
            'val_steps': val_steps,
            'val_losses': val_losses
        }

        with open(log_path, 'w') as f:
            json.dump(loss_data, f, indent=4)



def load_model_params(model_file_path):
            loaded_data = np.load(model_file_path, allow_pickle=True)
            params = {}
            for key in loaded_data.files:
                if 'multihead_params_' in key:
                    multihead_weights = []
                    for n in range(len(loaded_data[key])):
                        ar = jax.tree_util.tree_map(jnp.asarray, loaded_data[key][n])
                        multihead_weights.append(ar)
                    params.update({key: multihead_weights})
                else:
                    params.update({key: jnp.asarray(loaded_data[key])})
            return params