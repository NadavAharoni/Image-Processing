import numpy as np


def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:
    """
    Warp an image by applying scaling then rotation, both centered on the image.

    Parameters
    ----------
    image     : (H, W, C) uint8 array
    angle_deg : counter-clockwise rotation in degrees
    scale_x   : horizontal scale factor (>1 zooms in, <1 zooms out)
    scale_y   : vertical scale factor

    Returns
    -------
    output : same shape and dtype as image; pixels outside the source are black
    """
    H, W, C = image.shape

    # ------------------------------------------------------------------ #
    # 1. Image center — the pivot point for both rotation and scaling      #
    # ------------------------------------------------------------------ #
    cx, cy = W / 2.0, H / 2.0

    # ------------------------------------------------------------------ #
    # 2. Rotation matrix (counter-clockwise, in homogeneous coordinates)  #
    # ------------------------------------------------------------------ #
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ])

    # ------------------------------------------------------------------ #
    # 3. Scaling matrix                                                    #
    # ------------------------------------------------------------------ #
    S = np.array([
        [scale_x, 0,       0],
        [0,       scale_y, 0],
        [0,       0,       1]
    ])

    # ------------------------------------------------------------------ #
    # 4. Compose the full affine transform M (forward: input → output)    #
    #                                                                      #
    #    We want scale and rotation to happen around the image center, so  #
    #    the pipeline is:                                                   #
    #      (a) translate center to origin                                  #
    #      (b) scale                                                        #
    #      (c) rotate                                                       #
    #      (d) translate origin back to center                             #
    # ------------------------------------------------------------------ #
    T_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T_back       = np.array([[1, 0,  cx], [0, 1,  cy], [0, 0, 1]])

    M = T_back @ R @ S @ T_to_origin

    # ------------------------------------------------------------------ #
    # 5. Inverse transform (output → input, i.e. backward mapping)        #
    #                                                                      #
    #    Instead of asking "where does each source pixel go?" (forward),   #
    #    we ask "where did each output pixel come from?" (backward).       #
    #    Backward mapping is preferred because it visits every output       #
    #    pixel exactly once and avoids holes.                               #
    # ------------------------------------------------------------------ #
    M_inv = np.linalg.inv(M)

    # ------------------------------------------------------------------ #
    # 6. Build a grid of every output pixel coordinate — all at once      #
    #                                                                      #
    #    grid_x[r, c] = c  (column index)                                 #
    #    grid_y[r, c] = r  (row index)                                    #
    # ------------------------------------------------------------------ #
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))  # (H, W) each

    # Stack into homogeneous (x, y, 1) column vectors: shape (3, H*W)
    ones   = np.ones(H * W)
    coords = np.stack([grid_x.ravel(), grid_y.ravel(), ones])  # (3, H*W)

    # Apply the inverse transform to every output pixel simultaneously
    src_coords = M_inv @ coords              # (3, H*W)
    src_x = src_coords[0].reshape(H, W)     # fractional source column
    src_y = src_coords[1].reshape(H, W)     # fractional source row

    # ------------------------------------------------------------------ #
    # 7. Bilinear interpolation                                            #
    #                                                                      #
    #    Each src_x/src_y is usually non-integer, so we blend the four    #
    #    neighbouring pixels weighted by distance.                          #
    #                                                                      #
    #         x0    x1                                                      #
    #    y0   I00 -- I01                                                    #
    #          |      |                                                     #
    #    y1   I10 -- I11                                                    #
    # ------------------------------------------------------------------ #
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    # Fractional offset within the cell — used as interpolation weights
    dx = (src_x - x0)[:, :, np.newaxis]   # shape (H, W, 1) → broadcasts over C
    dy = (src_y - y0)[:, :, np.newaxis]

    # Clip indices to valid range before indexing (out-of-bounds pixels
    # are handled by the mask below, not by clamping values)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    # Gather the four neighbours (each has shape (H, W, C))
    I00 = image[y0c, x0c]   # top-left
    I01 = image[y0c, x1c]   # top-right
    I10 = image[y1c, x0c]   # bottom-left
    I11 = image[y1c, x1c]   # bottom-right

    # Weighted blend
    output = (I00 * (1 - dx) * (1 - dy) +
              I01 *      dx  * (1 - dy) +
              I10 * (1 - dx) *      dy  +
              I11 *      dx  *      dy)

    # Zero out pixels that mapped outside the source image
    valid = (src_x >= 0) & (src_x < W) & (src_y >= 0) & (src_y < H)
    output[~valid] = 0

    return output.astype(image.dtype)
