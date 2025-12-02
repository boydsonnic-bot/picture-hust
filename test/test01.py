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
    
    if len(contour_info) == 3:
        _, contours, _ = contour_info
        return
    else:
        contours , _ = contour_info
    

    count = 0
    result = img.copy()
   
    for contour in contours :
        area = cv2.contourArea(contour)
        if  area > 0 :
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(result, f'Area: {int(area)}', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            count += 1
            print ('contour found')
            print(f'Contour {count}: Area = {area}')

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
    
