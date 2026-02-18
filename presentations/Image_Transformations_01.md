# Image Transformations & Interpolation (01)

------------------------------------------------------------------------

## Matrix Multiplication (Reminder)

If:

$$
A \in \mathbb{R}^{m \times n}, \quad
B \in \mathbb{R}^{n \times k}
$$

Then:

$$
(AB)_{ij} = \sum_{r=1}^{n} A_{ir} B_{rj}
$$

Condition:\
Columns of $A$ must equal rows of $B$.

------------------------------------------------------------------------

## Example

$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

$$
v =
\begin{bmatrix}
5 \\
6
\end{bmatrix}
$$

$$
Av =
\begin{bmatrix}
17 \\
39
\end{bmatrix}
$$

------------------------------------------------------------------------

## Matrix Multiplication in NumPy

``` python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

v = np.array([5, 6])

result = A @ v
print(result)
```

------------------------------------------------------------------------

# Scaling

Uniform scaling:

$$
S =
\begin{bmatrix}
s & 0 \\
0 & s
\end{bmatrix}
$$

Example ($s = 2$):

$$
(1, 3) \rightarrow (2, 6)
$$

Non-uniform scaling:

$$
S =
\begin{bmatrix}
s_x & 0 \\
0 & s_y
\end{bmatrix}
$$

------------------------------------------------------------------------

# Rotation

Counter-clockwise rotation:

$$
R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

Example ($90^\circ$):

$$
R =
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

$$
(1,0) \rightarrow (0,1)
$$

------------------------------------------------------------------------

# Composition of Transformations

If we first scale and then rotate:

$$
x' = R S x
$$

Important:

-   The transformation closest to the vector is applied first.
-   Matrix multiplication is **not commutative**.

------------------------------------------------------------------------

# Order Matters

$$
RS \neq SR
$$



