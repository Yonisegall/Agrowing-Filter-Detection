import cv2
import numpy as np

def align_to_reference(ref_img, img_to_align):
    # Convert to grayscale if not already
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    # Convert to float32 (required for ECC)
    ref_gray = ref_gray.astype(np.float32)
    align_gray = align_gray.astype(np.float32)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

    try:
        cc, warp_matrix = cv2.findTransformECC(ref_gray, align_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        aligned = cv2.warpAffine(img_to_align, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except cv2.error as e:
        print("ECC failed:", e)
        return img_to_align  # fallback
