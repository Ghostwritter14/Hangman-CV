import cv2


class BubbleOverlay:
    def __init__(self, bubble_image_path):
        self.bubble_img = cv2.imread(bubble_image_path, cv2.IMREAD_UNCHANGED)
        if self.bubble_img is None:
            raise FileNotFoundError("The bubble image could not be loaded. Please check the file path.")

    def resize_and_overlay_bubble(self, frame, face_coordinates, mistake_count):
        for (face_x, face_y, face_width, face_height) in face_coordinates:
            # Calculate bubble size based on face dimensions and mistakes
            resized_bubble, adjusted_face_x, adjusted_face_y = self._resize_bubble(
                face_x, face_y, face_width, face_height, mistake_count, frame
            )

            frame = self._overlay_resized_bubble(frame, resized_bubble, adjusted_face_x, adjusted_face_y)

        return frame

    def _resize_bubble(self, face_x, face_y, face_width, face_height, mistake_count, frame):
        scale_factor = 1 + (mistake_count * 0.2)
        resized_width = int(face_width * scale_factor)
        resized_height = int(face_height * scale_factor)

        # Ensure the resized bubble fits within the frame dimensions
        resized_width = min(resized_width, frame.shape[1] - face_x)
        resized_height = min(resized_height, frame.shape[0] - face_y)

        # Resize the bubble to the new width and height
        resized_bubble = cv2.resize(self.bubble_img, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        # Adjust the x, y position of the face to account for bubble offsets
        adjusted_face_x, adjusted_face_y = self._adjust_face_position(face_x, face_y, face_width, face_height,
                                                                      resized_width, resized_height)

        return resized_bubble, adjusted_face_x, adjusted_face_y

    def _adjust_face_position(self, face_x, face_y, face_width, face_height, resized_width, resized_height):
        x_offset = (resized_width - face_width) // 2
        y_offset = (resized_height - face_height) // 2

        adjusted_face_x = max(face_x - x_offset, 0)
        adjusted_face_y = max(face_y - y_offset, 0)

        return adjusted_face_x, adjusted_face_y

    def _overlay_resized_bubble(self, frame, resized_bubble, face_x, face_y):
        x_end = min(face_x + resized_bubble.shape[1], frame.shape[1])
        y_end = min(face_y + resized_bubble.shape[0], frame.shape[0])

        actual_bubble_width = x_end - face_x
        actual_bubble_height = y_end - face_y

        # Resize the bubble again if required
        if (resized_bubble.shape[1] != actual_bubble_width) or (resized_bubble.shape[0] != actual_bubble_height):
            resized_bubble = cv2.resize(resized_bubble, (actual_bubble_width, actual_bubble_height),
                                        interpolation=cv2.INTER_AREA)
        # Create the masks for overlay
        alpha_mask, inverse_alpha_mask = self._create_masks(resized_bubble)

        roi = frame[face_y:y_end, face_x:x_end]

        overlayed_roi = self._apply_mask_and_overlay(roi, resized_bubble, alpha_mask, inverse_alpha_mask)

        frame[face_y:y_end, face_x:x_end] = overlayed_roi

        return frame

    @staticmethod
    def _create_masks(resized_bubble):
        # Create alpha mask and its inverse from the bubble image's alpha channel
        alpha_mask = resized_bubble[:, :, 3]
        inverse_alpha_mask = cv2.bitwise_not(alpha_mask)

        # Merge to create 3-channel masks
        alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])
        inverse_alpha_mask = cv2.merge([inverse_alpha_mask, inverse_alpha_mask, inverse_alpha_mask])

        return alpha_mask.astype('uint8'), inverse_alpha_mask.astype('uint8')

    @staticmethod
    def _apply_mask_and_overlay(roi, resized_bubble, alpha_mask, inverse_alpha_mask):
        # Black-out the area of the bubble in the ROI using the inverse mask
        background = cv2.bitwise_and(roi, inverse_alpha_mask)

        # Extract bubble region from the resized bubble
        bubble_region = cv2.bitwise_and(resized_bubble[:, :, :3], alpha_mask)

        return cv2.add(background, bubble_region)



