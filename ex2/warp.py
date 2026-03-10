import numpy as np

def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:

    H, W, C = image.shape

    # TODO:
    # 1. Compute center (cx, cy)
    # 2. Build rotation matrix
    # 3. Build scaling matrix
    # 4. Compose full affine matrix
    # 5. Compute inverse
    # 6. For each output pixel:
    #       - map center coordinate backward
    #       - interpolate
    # 7. Return output image

    print(scale_x, scale_y, angle_deg)
    # raise NotImplementedError()