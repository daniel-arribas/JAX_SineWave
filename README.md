# JAX_SineWave

This pryect is a tutorial of NNs in native JAX consinsting in aproximating the sine function using MLP architecture.

## Structure of the proyect

```txt
JAX_SineWave                                                                       
    ├── config.json                                                                    
    ├── dataset.py                                                                     
    ├── loss.py                                                                        
    ├── model.py                                                                       
    ├── README.md                                                                      
    ├── test.py                                                                        
    └── train.py       

```

## Model definition (model.py)

The model is defined by the following functions in the model.py file.

### Parameter initializaion

```python
def init_params(key, layers):
    '''
    Initiolization of the weights and bias for each MLP layer.
    layers: List with dimensions example[1, 32, 32, 1]
    '''
    keys = jax.random.split(key,len(layers)-1)
    params = []

    for in_dim, out_dim, k in zip(layers[:-1], layers[1:], keys):
        #Xavier/Glorot initialization
        limit = jnp.sqrt(60/(in_dim + out_dim))
        w = jax.random.uniform(k, (in_dim, out_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim,))
        
        params.append({'w': w, 'b': b})

    return params
```

### NN evalutaion

```python
def forward(params, x):
    '''
    Evaluates the NN for a given x input
    '''

    # Processing hidden layers with senoidal activation
    for layer in params [:-1]:
        x = jnp.dot(x, layer['w']) + layer['b']
        x = jnp.sin(x) # Activacion periodica

    # output layer
    final_layer = params[-1]
    out = jnp.dot(x, final_layer['w']) + final_layer['b']

    return out
```





## Dataset management (dataset.py)

### Libraries
We import jax and jax.numpy
```python
import jax
import jax.numpy as jnp
```

### Full dataset creation

We generate a full dataset (without persitence) for testing and inference.

```python
def generate_full_dataset(
    num_samples=1000, x_min=-2 * jnp.pi, x_max=2 * jnp.pi
):
  
  # Generates equidistant points and converts them to shape(N, 1)
  x = jnp.linspace(x_min, x_max, num_samples).reshape(-1, 1)

  # Applies sin() 
  y = jnp.sin(x)

  return x, y
```

### Random batch generation for training

A random batch is generated for better resource use and also for generalization as the values each step of the training
will be different due to this.

```python
def get_batch(
    key, batch_size=32, x_min=-2 * jnp.pi, x_max=2 * jnp.pi, noise_std=0.0
):
  """.
  Args:
      key: PRNGKey de JAX for pure random sampling.
      batch_size: number of samples in the batch.
      x_min, x_max: Input domain limits.
      noise_std: Gaussian noise stdev (0.0 = no noise).

  Returns:
      x, y: Tensors with shape: (batch_size, 1).
  """
  # We divide the key in two parts to separate sampling of "x" from noise
  key_x, key_noise = jax.random.split(key)

  # We sample batch_size random homogeneously distributed points in [x_min, x_max]
  x = jax.random.uniform(
      key_x, shape=(batch_size, 1), minval=x_min, maxval=x_max
  )

  # We calculate gaussian noise and the label y = sin(x) + noise
  noise = jax.random.normal(key_noise, shape=(batch_size, 1)) * noise_std
  y = jnp.sin(x) + noise

  return x, y
```

### Verification code

This code verifies that the python file works.

```python
if __name__ == "__main__":
  # 1. Base random key
  key = jax.random.PRNGKey(42)

  # 2. Training batch generator is tested
  x_batch, y_batch = get_batch(key, batch_size=10)
  print("Shape of  Batch X (Training):", x_batch.shape)
  print("Shape of Batch Y (Training):", y_batch.shape)

  # 3. Testing dataset generator is tested
  x_eval, y_eval = generate_full_dataset(num_samples=100)
  print("Shape of Dataset X (Testing):", x_eval.shape)
  print("Shape of Dataset Y (Testing):", y_eval.shape)
```


## Loss function (loss.py)

## Training (train.py)

## Testing (test.py)

