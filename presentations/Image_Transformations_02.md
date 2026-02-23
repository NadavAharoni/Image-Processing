# Image Transformations & Interpolation (02)

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
