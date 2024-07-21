import cv2

# Load the Haar cascade classifier for detecting Russian license plates
nPlateCascade = cv2.CascadeClassifier("C:/Users/abish/Music/123/Number-Plate-Detector-main/Number-Plate-Detector-main/haarcascade_russian_plate_number.xml")

# Minimum area for a detected plate
minArea = 500

# Colors for drawing rectangles and text
color1 = (255, 0, 255)
color2 = (0, 255, 0)

# Initialize video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, you can change it to the desired camera index

# Loop to continuously capture frames from the camera feed
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    if not ret:
        break  # If no frame is captured, break the loop

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect potential number plates in the frame
    numberPlates = nPlateCascade.detectMultiScale(gray, 1.1, 4)

    # Iterate over the detected plates
    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            # Draw rectangle around the detected plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), color1, 2)
            # Add text label "Number Plate" above the plate
            cv2.putText(frame, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color1, 2)
            # Extract the detected number plate region
            imgNumberPlate = frame[y:y + h, x:x + w]
            # Display the extracted number plate
            cv2.imshow("Number Plate", imgNumberPlate)

    # Display the original frame with detected plates
    cv2.imshow("Test Screen", frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
