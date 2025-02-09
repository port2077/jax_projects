import time

import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import datasets, transforms
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from flax import nnx
import optax

print(jax.devices())

class Model(nnx.Module):

    def __init__(self, rngs : nnx.Rngs):
        super().__init__()

        ## jax expects image dimensions in NCHW order {batch size, channels, height, width}
        self.conv1 = nnx.Conv(
            in_features = 1,
            out_features = 8,
            kernel_size = (3,3),
            strides = 1,
            padding = 'SAME',
            rngs = rngs
        )
        self.conv2 = nnx.Conv(
            in_features= 8,
            out_features = 16,
            kernel_size = (3,3),
            strides = 1,
            padding = 'SAME',
            rngs = rngs
        )
        self.conv3 = nnx.Conv(
            in_features= 16,
            out_features = 32,
            kernel_size = (3,3),
            strides = 1,
            padding = 'SAME',
            rngs = rngs
        )
        
        self.dropout = nnx.Dropout(rate=0.5, rngs = rngs)
        self.linear1 = nnx.Linear(32*14*14,10, rngs = rngs)
        

    
    def __call__(self,x):

        #print(f'Input tensor shape : {x.shape}')
        x = jnp.transpose(x,[0,2,3,1])
        #print(f'Input tensor shape : {x.shape}')
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = self.dropout(x)
        x = nnx.relu(self.conv3(x))
        x = nnx.max_pool(x,
            window_shape = (2,2),
            strides = (2,2),
            padding= 'SAME'
            )
        #print(f'Tensor shape after maxpool: {x.shape}')
        ## x = jax.lax.collapse(x,0) this does not work for flattening
        x = x.reshape((x.shape[0],-1))
        x = self.linear1(x)
        ## jax loss softmax cross entropy handles sigmoid on its own
        # x = nnx.sigmoid(x)

        return x
    

## collate fucntion that takes torch arrays 
## and convert them to jax,array datatype

def jax_collate(batch):
    return tree_map(jnp.asarray,default_collate(batch))

## define the loss function which will return the numerical loss value


@nnx.jit
def train(epochs):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data',train=True,download=True,transform=transform)
    test_dataset = datasets.MNIST('data',train=False,download=True,transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 32,
        shuffle = True,
        collate_fn= jax_collate
    ) 

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=jax_collate
    )

    model = Model(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=1e-3)
    )

    ## since jax is based on functional programming , loss and grads
    ## are defined and/or calculated as functions which returns
    ## the numerical value of the loss/grad which can then be used
    ## to update the weights
    start_time = time.time()
    for epoch in range(5):
        model.train()
        loss_val = 0
        correct = 0
        total = 0

        for batch_idx,(data,target) in enumerate(train_dataloader):

            def loss_fx(model):
                #print(f'Input tensor shape: {data.shape}')
                output = model(data)
                #print(f'Output tensor shape: {output.shape} Output: {output}')
                # predicted = jnp.argmax(output, 1)  # Get the predicted class
                # total += target.size(0)  # Add batch size
                # correct += (predicted == target).sum()
                # get the loss value
                # add comment for integer label / mean
                loss = optax.softmax_cross_entropy_with_integer_labels(output,target).mean()
                return loss

            ## get the grad fx that needs to calculate the gradients
            ## the jax nnx value_and_grad takes as input a function
            ## and returns the same function extended (?)
            
            grad_fx = nnx.value_and_grad(loss_fx)
            ## get the numerical gradient values
            loss,grads = grad_fx(model)
            ## use grads to update the NN weights
            optimizer.update(grads)

        #accuracy = 100 * correct / total
        print(f'Epoch: {epoch}/{5-1}')
        #print(f'Epoch: {epoch}/{5-1}  Accuracy: {accuracy:.2f}%')
    end_time = time.time()
    print(f'Train time: {end_time - start_time:.2f} seconds')

    model.eval()
    num_of_correct = 0
    total = 0
    for batch_idx,(data,target) in enumerate(test_dataloader):
        output = model(data)
        predicted = jnp.argmax(output, 1)  # Get the predicted class
        total += target.shape[0]  # Add batch size
        num_of_correct += (predicted == target).sum()
    print(f'Test Dataset Accuracy: {num_of_correct/total}')



            
if __name__ == "__main__" :

    start = time.time()
    train(5)
    end = time.time()
    print(f'Runtime:')
    print(f'Total runtime: {end - start:.2f} seconds')












