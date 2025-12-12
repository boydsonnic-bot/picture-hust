import cv2
import numpy as np
import argparse
import os
from numpy.typing import NDArray
from typing import Any
def parse_args():
    p = argparse.ArgumentParser(description='Contour detection v1.2 (basic + save)')
    p.add_argument('--image', type=str, default='image.png', help='input image path')
    p.add_argument('--save', type=str, default='result_contours.png', help='output image path')
    return p.parse_args()
Contour = NDArray[np.generic]
def feature_extract(contour: Any) -> list[float]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = float((4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0)
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w / h) if h > 0 else 0.0

    hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
    hu_list = [float(hm) for hm in hu_moments]

    return [
        float(area),
        float(perimeter),
        float(circularity),
        float(aspect_ratio)
    ] + hu_list

def main():
    arg = parse_args()
    img = cv2.imread(arg.image)
    if img is None:
        print('failed to load image:', arg.image)
        return

    grays = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    gray = clahe.apply(grays)
    blured = cv2.bilateralFilter(gray, 0, 75, 75)
    adaptive_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mo = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)



    contour_info = cv2.findContours(mo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours: list[Any] = list(contour_info[-2])

    result = img.copy()
    count = 0
    features_list: list[list[float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        feature = feature_extract(contour)
        features_list.append(feature)

        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            count += 1
            print('Contour found:', count, "- Area =", area)


    print(f'[INFO] total contours found: {count}')

    cv2.imshow('Contours', result)
    cv2.imshow('Gray', gray)
    cv2.imshow('Blurred (5x5)', blured)
    cv2.imshow('Adaptive Thresh', adaptive_thresh)
    cv2.imshow('Morphological Opening', mo) 
  
    if arg.save:
        cv2.imwrite(arg.save, result)
        print(f'[INFO] Saved: {os.path.abspath(arg.save)}')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
