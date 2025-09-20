# !curl -O https://raw.githubusercontent.com/h4pZ/rose-pine-matplotlib/refs/heads/main/themes/rose-pine-moon.mplstyle

import os
from pathlib import Path
import matplotlib.pyplot as plt

style_file_path = "./rose-pine-moon.mplstyle"
plt.style.use(style_file_path)

path_to_file = "prngs.txt"
output_path = Path("dist_plot.png")


def read_numbers(file_path: str) -> list[int]:
    """Read random numbers from a file, one per line."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    nums = []
    with Path(file_path).open("r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    nums.append(int(stripped))
                except ValueError:
                    raise ValueError(f"Invalid number in file: {stripped}")
    return nums


def plot_numbers(nums: list[int], output_path: str) -> None:
    """Plot the PRNG numbers as a scatter plot and save to file."""
    os.makedirs(Path(output_path).parent, exist_ok=True)
    # Removed the second plt.style.use() call as it's already applied globally
    plt.figure(figsize=(10, 8))
    plt.scatter(
        range(len(nums)), nums, s=1, color="#e0def4"
    )  # Changed color to a hex code from the style
    plt.title("(clib) PRNG Distribution (10K)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to: {output_path}")


try:
    nums = read_numbers(path_to_file)
except Exception as e:
    print(f"Error reading numbers: {e}")
    raise

plot_numbers(nums, output_path)
