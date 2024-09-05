import cv2
import numpy as np


class WinGame:
    def __init__(self, crown_path, firework_path):
        self.crown_image = cv2.imread(crown_path, cv2.IMREAD_UNCHANGED)
        self.firework_image = cv2.imread(firework_path, cv2.IMREAD_UNCHANGED)
        self.cap = cv2.VideoCapture(0)
        if self.crown_image is None or self.firework_image is None:
            raise FileNotFoundError("One or more image assets could not be loaded.")

    def place_crown(self, frame, face_coordinates):
        x, y, w, h = face_coordinates

        # Resize the crown image to fit the width `w` and height `h // 2`
        crown_resized = cv2.resize(self.crown_image, (w, h // 2), interpolation=cv2.INTER_AREA)

        # Calculate the position to place the crown, accounting for the height of the crown
        crown_x = x
        crown_y = y - h // 2  # crown placed over the head

        crown_alpha = crown_resized[:, :, 3] / 255.0
        inverted_alpha = 1 - crown_alpha

        crown_y = max(crown_y, 0)

        for c in range(0, 3):
            frame[crown_y:crown_y + h // 2, crown_x:crown_x + w, c] = (
                    crown_alpha * crown_resized[:, :, c] +
                    inverted_alpha * frame[crown_y:crown_y + h // 2, crown_x:crown_x + w, c]
            )
        return frame

    def show_fireworks(self, frame, firework_resized, step, total_steps):
        alpha = step / total_steps
        h, w = firework_resized.shape[:2]

        positions = [
            (0, 0),
            (frame.shape[1] - w, 0),
            (0, frame.shape[0] - h),
            (frame.shape[1] - w, frame.shape[0] - h)
        ]

        for pos in positions:
            firework_alpha = firework_resized[:, :, 3] / 255.0
            firework_image = firework_resized[:, :, :3]

            # Use the alpha channel to blend the fireworks with the frame
            for c in range(0, 3):
                frame[pos[1]:pos[1] + h, pos[0]:pos[0] + w, c] = (
                        firework_alpha * firework_image[:, :, c] * alpha +
                        frame[pos[1]:pos[1] + h, pos[0]:pos[0] + w, c] * (1 - (firework_alpha * alpha))
                )

        return frame

    def transition_to_win_screen(self, cap, face_detector):
        steps = 60  # Increase the number of steps to slow down the fireworks
        delay = 100  # Increase the delay to slow down the fireworks (milliseconds)

        # Get the size of the video feed
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return
        feed_height, feed_width = frame.shape[:2]

        # Resize the fireworks image to fit the corners of the frame
        firework_resized = cv2.resize(self.firework_image, (feed_width // 4, feed_height // 4))

        for i in range(steps):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply fireworks effect
            frame = self.show_fireworks(frame, firework_resized, i, steps)

            # Check for face detection and apply crown if a face is detected
            faces = face_detector.detect_faces(frame)
            if faces is not None and len(faces) > 0:
                frame = self.place_crown(frame, faces[0])

            cv2.imshow('Frame', frame)
            cv2.waitKey(delay)

        cv2.waitKey(0)

    def end_game(self):
        cv2.destroyAllWindows()
        self.cap.release()
