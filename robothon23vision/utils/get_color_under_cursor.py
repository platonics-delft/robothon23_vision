import cv2
import numpy as np

hsv_codes = []
 
 

def get_color_under_cursor(img_name: str) -> None:
    img = cv2.imread(img_name)
    def mouseRGB(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
            colorsB = img[y,x,0]
            colorsG = img[y,x,1]
            colorsR = img[y,x,2]
            colors = img[y,x]
            hsv = cv2.cvtColor(np.uint8([[img[y,x]]]), cv2.COLOR_BGR2HSV)
            hsv_codes.append(hsv)
            print ("HSV : " ,hsv)
            #print("Red: ",colorsR)
            #print("Green: ",colorsG)
            #print("Blue: ",colorsB)
            #print("BRG Format: ",colors)
            print("Coordinates of pixel: X: ",x,"Y: ",y)
    cv2.namedWindow('mouseRGB')
    cv2.setMouseCallback('mouseRGB',mouseRGB)
     
#Do until esc pressed
    while(1):
        cv2.imshow('mouseRGB',img)
        if cv2.waitKey(20) & 0xFF == 27:
            print(f"lower = np.array({np.min(np.array(hsv_codes), axis=0)[0][0]})")
            print(f"upper= np.array({np.max(np.array(hsv_codes), axis=0)[0][0]})")
            break
#if esc pressed, finish.
    cv2.destroyAllWindows()
