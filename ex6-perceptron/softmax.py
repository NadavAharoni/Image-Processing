import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

examples = [
    [1, 2, 3],
    [-1000, 0, 1000],
    [0.1, 0.2, 0.3],
    [5, 5, 5],
]

for values in examples:
    result = softmax(values)
    print(f"Input:  {values}")
    print(f"Output: {np.round(result, 4)}")
    print(f"Sum:    {result.sum():.6f}")
    print()
