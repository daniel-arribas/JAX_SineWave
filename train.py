import jax
import jax.numpy as jnp
from dataset import get_batch
from loss import loss_and_grad_fn
from model import init_params

# ==============================================================================
# 1. JIT-COMPILED UPDATE STEP
# ==============================================================================


@jax.jit
def train_step(params, x_batch, y_batch, learning_rate):
  """Performs a single optimization step:

  1. Computes loss and gradients.
  2. Updates parameters using Stochastic Gradient Descent (SGD).
  3. Returns updated parameters and the current loss value.
  """
  # Compute loss and gradients using the function from loss.py
  loss_val, grads = loss_and_grad_fn(params, x_batch, y_batch)

  # Update parameters using SGD: new_p = p - lr * grad
  updated_params = jax.tree.map(
      lambda p, g: p - learning_rate * g, params, grads
  )

  return updated_params, loss_val


# ==============================================================================
# 2. MAIN TRAINING LOOP
# ==============================================================================


def train():
  # Hyperparameters
  layer_sizes = [1, 64, 64, 1]
  learning_rate = 0.05
  num_epochs = 5000
  batch_size = 64
  print_every = 500

  # PRNG Key setup
  key = jax.random.PRNGKey(42)
  key, init_key = jax.random.split(key)

  # 1. Initialize model parameters
  params = init_params(init_key, layer_sizes)
  print("Model initialized. Starting training...\n")

  # 2. Training Loop
  for epoch in range(1, num_epochs + 1):
    # Split key to generate a new random batch deterministically
    key, batch_key = jax.random.split(key)
    x_batch, y_batch = get_batch(batch_key, batch_size=batch_size)

    # Perform the compiled update step
    params, loss_val = train_step(params, x_batch, y_batch, learning_rate)

    # Monitor progress
    if epoch % print_every == 0 or epoch == 1:
      print(f"Epoch {epoch:4d} / {num_epochs} | Loss: {loss_val:.6f}")

  print("\nTraining completed successfully!")
  return params


if __name__ == "__main__":
  trained_params = train()