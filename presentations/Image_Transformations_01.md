# Image Transformations & Interpolation

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

Demonstrate numerically in class.

------------------------------------------------------------------------

# Why Translation Cannot Be $2 \times 2$

Assume:

$$
Ax = x + t
$$

If $x = 0$:

$$
A0 = 0
$$

But:

$$
0 + t = t \neq 0
$$

Translation is not linear.

------------------------------------------------------------------------

# Homogeneous Coordinates

$$
(x, y) \rightarrow (x, y, 1)
$$

Translation matrix:

$$
T =
\begin{bmatrix}
1 & 0 & a \\
0 & 1 & b \\
0 & 0 & 1
\end{bmatrix}
$$

Now we can combine:

$$
T R S
$$

------------------------------------------------------------------------

# Forward vs Backward Mapping

Forward mapping:

-   For each source pixel compute destination
-   Problem: holes

Backward mapping:

-   For each destination pixel compute source
-   Use inverse transform

------------------------------------------------------------------------

# Nearest Neighbor

-   Take closest pixel
-   Fast
-   Blocky

------------------------------------------------------------------------

# Bilinear Interpolation

$$
I(x,y) =
(1-\alpha)(1-\beta) I_{00}
+ \alpha(1-\beta) I_{10}
+ (1-\alpha)\beta I_{01}
+ \alpha\beta I_{11}
$$

Uses 4 neighbors.

------------------------------------------------------------------------

# Bicubic Interpolation

-   Uses 16 neighbors
-   Smoother
-   More computationally expensive
