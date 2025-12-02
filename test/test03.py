import cv2
import numpy as np 
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description='Contour detection v1.2 (basic + save)')
    p.add_argument('--image', type=str, default='image.png', help='input image path')
    p.add_argument('--save', type=str, default='result_contours.png', help='output image path')
    return p.parse_args()


def main():
    arg = parse_args()
    img = cv2.imread(arg.image)
    if img is None:
        print('failed to load image')
        return
    

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 0)
    var ,thresh = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # update for different OpenCV versions
    contours = contour_info[-2]
    
    count = 0
    result = img.copy()
   
    for contour in contours:

    # Bỏ contour nhỏ (nhiễu)
        if cv2.contourArea(contour) < 100:
            continue

        # --- Geometric features ---
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        cv2.drawContours(result, [contour], -1, (0,255,0), 2)
        cv2.rectangle(result, (x,y), (x+w,y+h), (0,0,255), 2)
        # --- Hu Moments ---
        M = cv2.moments(contour)
        if M['m00'] != 0:
            hu = cv2.HuMoments(M).flatten()
            hu_log = [-np.sign(h) * np.log10(abs(h) + 1e-10) for h in hu]
        else:
            hu_log = [0]*7

        print("\n--- Contour ---")
        print(f"Area: {area:.2f}")
        print(f"Perimeter: {perimeter:.2f}")
        print(f"Circularity: {circularity:.3f}")
        print(f"Aspect Ratio: {aspect_ratio:.3f}")
        print("Hu Moments:", hu_log)

    # Optional: Hiển thị trực quan
            


    # print('total contours found:', count)
    # print('threshold value otsu:', var)

    print(f'[INFO] otsu threshold value: {var}')
    print(f'[INFO] total contours found: {count}')

    cv2.imshow('Contours', result)
    cv2.imshow('Gray', gray)
    cv2.imshow('Blurred (5x5)', blured)
    cv2.imshow('Threshold (Otsu)', thresh)

    if arg.save:
        cv2.imwrite(arg.save, result)
        print(f'[INFO] Saved: {os.path.abspath(arg.save)}')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
