import cv2
import numpy as np


class GarbageDetector:
    def __init__(self, resize_dims=(640, 480), min_area=500,
                 aspect_ratio_range=(0.2, 5), solidity_thresh=0.2):
        self.resize_dims = resize_dims
        self.min_area = min_area
        self.aspect_ratio_range = aspect_ratio_range
        self.solidity_thresh = solidity_thresh

        # Pre-allocate kernels
        self.morph_kernel_small = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3))
        self.morph_kernel_large = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7))

    def preprocess_image(self, image):
        # Downscale for faster processing
        scale_factor = 0.5
        temp_dims = (
            int(self.resize_dims[0]*scale_factor), int(self.resize_dims[1]*scale_factor))
        resized = cv2.resize(image, temp_dims)
        resized = cv2.resize(resized, self.resize_dims,
                             interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Simpler, faster noise reduction
        blurred = cv2.medianBlur(gray, 5)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 5)

        # Fast morphology
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, self.morph_kernel_small)

        return resized, thresh

    def get_robust_color_masks(self, image, thresh_mask):
        # Convert only to HSV (faster than multiple color spaces)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Skip the median blur, use Gaussian instead (faster)
        hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # Create masks with pre-defined parameters
        paper_mask = cv2.inRange(hsv_blurred, np.array(
            [0, 0, 180]), np.array([180, 50, 255]))
        paper_mask = cv2.bitwise_and(paper_mask, thresh_mask)

        plastic_mask = cv2.inRange(hsv_blurred, np.array(
            [0, 20, 80]), np.array([180, 255, 255]))
        plastic_mask = cv2.bitwise_and(plastic_mask, thresh_mask)

        # Combine masks
        trash_mask = cv2.bitwise_or(paper_mask, plastic_mask)

        # Simplified morphology with fewer iterations
        trash_mask = cv2.morphologyEx(
            trash_mask, cv2.MORPH_CLOSE, self.morph_kernel_large, iterations=1)

        return trash_mask

    def filter_contours(self, contours):
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue

            # Skip solidity check for small contours (faster)
            if area < self.min_area * 2:
                valid_contours.append(cnt)
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area >= self.solidity_thresh:
                valid_contours.append(cnt)

        return valid_contours

    def detect(self, image):
        # Preprocess with optimized settings
        resized, thresh_mask = self.preprocess_image(image)
        result_image = resized.copy()

        # Get optimized masks
        trash_mask = self.get_robust_color_masks(resized, thresh_mask)

        # Use simpler contour approximation method (faster)
        contours, _ = cv2.findContours(
            trash_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours
        valid_contours = self.filter_contours(contours)

        # Annotate results
        detections = []
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append({
                'position': (x, y, w, h),
                'area': cv2.contourArea(cnt),
                'contour': cnt
            })
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(result_image, "Trash", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(result_image, f"Detected: {len(valid_contours)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result_image, {'trash': detections}, {'trash_mask': trash_mask}
