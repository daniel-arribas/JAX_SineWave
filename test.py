import matplotlib.pyplot as plt
import jax.numpy as jnp
from dataset import generate_full_dataset
from model import forward
from train import train


def evaluate_and_plot():
  # 1. Train the model to get the trained PyTree parameters
  params = train()

  # 2. Generate a dense evaluation dataset [-2pi, 2pi]
  x_test, y_test = generate_full_dataset(
      num_samples=1000, x_min=-2 * jnp.pi, x_max=2 * jnp.pi
  )

  # 3. Direct inference pass
  y_pred = forward(params, x_test)

  # 4. Compute Test MSE
  test_mse = jnp.mean((y_pred - y_test) ** 2)
  print(f"\nFinal Test MSE: {test_mse:.6f}")

  # 5. Plot True Sine vs Model Approximation
  plt.figure(figsize=(10, 5))
  plt.plot(
      x_test,
      y_test,
      label="Ground Truth: sin(x)",
      color="black",
      linewidth=2,
      linestyle="--",
  )
  plt.plot(
      x_test,
      y_pred,
      label="MLP Approximation (JAX)",
      color="red",
      linewidth=1.5,
  )
  plt.title("JAX MLP Sine Function Approximation")
  plt.xlabel("x")
  plt.ylabel("sin(x)")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()

  # Save figure to disk
  plt.savefig("sine_approximation.png", dpi=300)
  print("Plot saved successfully as 'sine_approximation.png'")


if __name__ == "__main__":
  evaluate_and_plot()