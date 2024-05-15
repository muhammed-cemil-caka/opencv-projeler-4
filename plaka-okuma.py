import cv2
import datetime

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    prev_plate = None
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        plates = plate_cascade.detectMultiScale(blurred, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        
        for (x, y, w, h) in plates:
            
            if prev_plate is None or (x, y, w, h) != prev_plate:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                plate_img = frame[y:y+h, x:x+w]
                
                cv2.imshow('Algılanan Plaka', plate_img)
                
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_plate.jpg"
                cv2.imwrite(filename, plate_img)
                
                prev_plate = (x, y, w, h)
        
        cv2.imshow('Plaka Tespiti', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
