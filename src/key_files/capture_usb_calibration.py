import cv2

camera_id = '/dev/video1'

#cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
cap = cv2.VideoCapture(camera_id)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow('Calibration Image', frame)
    
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC key to exit
        break
    elif key % 256 == 32:  # Space key to save the image
        img_name = f"./calibration_images/calibration_image_{count}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved!")
        count += 1

cap.release()
cv2.destroyAllWindows()

