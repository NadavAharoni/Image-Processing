import numpy as np


def main():
    # -----------------------------
    # Step 0: Create a small "image"
    # -----------------------------
    H, W = 4, 5

    # Simple grayscale image: value = row*10 + col
    img = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            img[r, c] = r * 10 + c

    print("Original image:")
    print(img)
    print()

    # -----------------------------
    # Step 1: Create coordinate grid
    # -----------------------------
    # np.indices returns two (H,W) arrays: rows[r,c]=r, cols[r,c]=c
    rows, cols = np.indices((H, W))

    print(rows)
    print()
    print(cols)
    print()
    
    # +0.5 shifts to pixel center (pixel [r,c] spans [r, r+1) x [c, c+1))
    x_dst = cols + 0.5
    y_dst = rows + 0.5

    print("Destination coordinates:")
    print(x_dst)
    print() 
    print(y_dst)
    print()

    # -----------------------------
    # Step 2: Build homogeneous coordinates
    # -----------------------------
    ones = np.ones_like(x_dst)
    # Stack into shape (3, H, W): each "column" is [x, y, 1] for one pixel
    coords = np.stack([x_dst, y_dst, ones], axis=0)   # (3, H, W)

    print(ones)
    print()
    print("Homogeneous coordinates:")
    print(coords)
    print()

    # Reshape to (3, H*W) so we can apply M_inv with a single matrix multiply
    coords_flat = coords.reshape(3, -1)               # (3, H*W)

    print("Flattened coordinates:")
    print(coords_flat)
    print()

    # -----------------------------
    # Step 3: Define transformation
    # Example: small translation
    # -----------------------------
    M_inv = np.array([
        [1, 0, -1],   # shift left
        [0, 1, -1],   # shift up
        [0, 0, 1]
    ])

    # Each column of coords_flat is one pixel; M_inv maps all of them at once
    src = M_inv @ coords_flat

    # -----------------------------
    # Step 4: Back to image shape
    # -----------------------------
    x_src = src[0].reshape(H, W)
    y_src = src[1].reshape(H, W)

    # Undo the +0.5 shift to get back to array index space
    r_src = y_src - 0.5
    c_src = x_src - 0.5

    # -----------------------------
    # Step 5: Nearest Neighbor
    # -----------------------------
    # Round to the nearest integer index (equivalent to picking the closest pixel)
    r_nn = np.round(r_src).astype(int)
    c_nn = np.round(c_src).astype(int)

    print("Nearest neighbor row indices:")
    print(r_nn)
    print()

    print("Nearest neighbor col indices:")
    print(c_nn)
    print()

    # -----------------------------
    # Step 6: Boundary mask
    # -----------------------------
    # Boolean (H,W) array: True where the source pixel falls inside the image
    valid = (
        (r_nn >= 0) & (r_nn < H) &
        (c_nn >= 0) & (c_nn < W)
    )

    print("Valid mask:")
    print(valid.astype(int))  # print as 0/1 for clarity
    print()

    # -----------------------------
    # Step 7: Build output image
    # -----------------------------
    output = np.zeros_like(img)

    # Boolean indexing: output[valid] selects only the True positions (as 1D).
    # r_nn[valid] / c_nn[valid] give matching 1D arrays of source indices.
    # img[r_nn[valid], c_nn[valid]] is fancy indexing: pairs are read element-wise.
    # The following line is equivalent to this loop (but vectorized):
    # for r in range(H):
    #     for c in range(W):
    #         if valid[r, c]:
    #             output[r, c] = img[r_nn[r, c], c_nn[r, c]]

    output[valid] = img[r_nn[valid], c_nn[valid]]

    print("Output image (Nearest Neighbor):")
    print(output)
    print()


if __name__ == "__main__":
    main()
