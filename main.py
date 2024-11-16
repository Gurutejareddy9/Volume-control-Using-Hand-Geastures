import cv2
import time
import numpy as np
import HandDetectingModule as htm  # Ensure this module exists
import subprocess  # For macOS volume control

# Camera resolution
wCam, hCam = 640, 480

# Initialize camera
cap = cv2.VideoCapture(0)  # Use default camera (0 for the default one)
cap.set(3, wCam)  # Set camera width
cap.set(4, hCam)  # Set camera height

pTime = 0  # Previous time for FPS calculation
detector = htm.handDetector(detectionCon=0.7, maxHands=1)  # Initialize hand detector

# Volume variables
volBar = 400  # Volume bar height
volPer = 0  # Volume percentage
area = 0  # Hand area
colorVol = (255, 0, 0)  # Initial color for volume bar

# Function to set system volume on macOS using AppleScript
def set_volume(volume_percentage):
    """Sets the system volume using AppleScript."""
    volume = int(volume_percentage)  # Use the volume percentage directly
    script = f"osascript -e 'set volume output volume {volume}'"
    subprocess.run(script, shell=True)

while True:
    success, img = cap.read()  # Capture frame from camera
    if not success:
        break

    # Find hand landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    
    if len(lmList) != 0:
        # Calculate area of the bounding box (hand size)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

        # If hand size is in a valid range, detect distance between thumb and index finger
        if 250 < area < 1000:
            length, img, lineInfo = detector.findDistance(4, 8, img)

            # Map the distance to volume (fine-tune the mapping ranges)
            volBar = np.interp(length, [50, 250], [400, 150])  # Map length to volume bar height
            volPer = np.interp(length, [50, 250], [0, 100])  # Map length to volume percentage

            # Make volume adjustment smoother
            smoothness = 10  # Experiment with this value to get smoother transitions
            volPer = smoothness * round(volPer / smoothness)

            # Check if pinky is down (indicates volume adjustment)
            fingers = detector.fingersUp()

            # If pinky is down, adjust the volume
            if not fingers[4]:
                set_volume(volPer)  # Set system volume on macOS using AppleScript
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)  # Green when volume is adjusted
            else:
                colorVol = (255, 0, 0)  # Red when pinky is up

    # Draw volume bar and percentage
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Display current system volume (simulated by the volume percentage)
    cv2.putText(img, f'Vol Set: {int(volPer)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, colorVol, 3)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Show the processed frame
    cv2.imshow("Img", img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
