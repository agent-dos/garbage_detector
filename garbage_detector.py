import cv2
import numpy as np

class GarbageDetector:
    def __init__(self, resize_dims=(640, 640), min_area=500, 
                 aspect_ratio_range=(0.2, 5), solidity_thresh=0.4):
        """
        Initialize detector with processing parameters and filtering thresholds.
        """
        self.resize_dims = resize_dims
        self.min_area = min_area
        self.aspect_ratio_range = aspect_ratio_range
        self.solidity_thresh = solidity_thresh

    def preprocess_image(self, image):
        """
        Improved preprocessing with adaptive thresholding and noise reduction.
        Returns resized image and threshold mask.
        """
        # Resize and convert to grayscale
        resized = cv2.resize(image, self.resize_dims)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction and adaptive thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        return resized, thresh

    def create_color_mask(self, hsv_image, lower, upper, thresh_mask):
        """Helper to create color masks combined with threshold mask"""
        mask = cv2.inRange(hsv_image, lower, upper)
        return cv2.bitwise_and(mask, thresh_mask)

    def filter_contours(self, contours):
        """Filter contours using area, aspect ratio, and shape characteristics"""
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # Aspect ratio filtering
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue

            # Solidity (contour area vs convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < self.solidity_thresh:
                continue

            valid_contours.append(cnt)
        return valid_contours

    def detect(self, image):
        """
        Enhanced detection with multiple filtering stages and improved color segmentation.
        Returns annotated image, detections, and masks.
        """
        # Preprocess image and create masks
        resized, thresh_mask = self.preprocess_image(image)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        result_image = resized.copy()

        # Color ranges (adjust based on your target environments)
        paper_mask = self.create_color_mask(hsv, 
            lower=np.array([0, 0, 200]), 
            upper=np.array([180, 40, 255]),
            thresh_mask=thresh_mask)

        plastic_mask = self.create_color_mask(hsv,
            lower=np.array([0, 30, 100]),
            upper=np.array([180, 255, 240]),
            thresh_mask=thresh_mask)

        # Combine masks and refine
        trash_mask = cv2.bitwise_or(paper_mask, plastic_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        trash_mask = cv2.morphologyEx(trash_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        trash_mask = cv2.morphologyEx(trash_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Contour detection and filtering
        contours, _ = cv2.findContours(trash_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

# Example usage:
# detector = GarbageDetector()
# result_img, detections, masks = detector.detect(your_image)