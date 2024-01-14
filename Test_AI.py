import cv2
import math
import time

# Path to Haarcascades
haarcascades_path = cv2.data.haarcascades
face_cascade_path = haarcascades_path + 'haarcascade_frontalface_default.xml'

# Initialize the cascade for face detection
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Define the video capture object
vid = cv2.VideoCapture(0)

# Set the known focal length of your camera (in pixels), replace it with the actual focal length of your camera
focal_length = 1000

# Real size of the object (in this case, the average face size in centimeters)
real_face_height_cm = 30.0

# Trajectory parameters
trajectory_speed = 5  # Speed of points (adjust as needed)

# Initialize the list of points
points = []

current_time = 0

while True:
    current_time += 1

    # Capture a video frame
    ret, frame = vid.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x_face, y_face, w_face, h_face) in faces:
        # Display a rectangle around the face
        cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)

        for point in points:
            x, y, remaining_frames, distance_cm = point

            if remaining_frames > 0:
                # Calculate the angle between the current position of the point and the center of the face
                angle = math.atan2(y_face - y, x_face - x)

                # Calculate the new position of the point considering the angle and speed
                x_new = x + trajectory_speed * math.cos(angle)
                y_new = y + trajectory_speed * math.sin(angle)

                # Display the point
                cv2.circle(frame, (int(x_new), int(y_new)), 3, (0, 0, 255), -1)

                # Update coordinates and distance
                points[points.index(point)] = (x_new, y_new, remaining_frames - 1, distance_cm)

                # Print point coordinates to the console
                print(
                    f"Point coordinates: ({int(x_new)}, {int(y_new)}), Remaining frames: {remaining_frames}, Distance: {distance_cm:.2f} cm")

                # If the point reaches the center of the face, remove it
                if 0 <= x_new <= w_face and 0 <= y_new <= h_face:
                    points[points.index(point)] = (x_new, y_new, 0, distance_cm)  # Set remaining_frames to 0

    # Remove old points
    points = [point for point in points if point[2] > 0]

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close the window
vid.release()
cv2.destroyAllWindows()
