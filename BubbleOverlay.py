import cv2


class BubbleOverlay:
    def __init__(self, bubble_image_path):
        # Load the bubble image with transparency
        self.bubble = cv2.imread(bubble_image_path, cv2.IMREAD_UNCHANGED)
        if self.bubble is None:
            raise FileNotFoundError("The bubble image could not be loaded. Please check the file path.")

    def resize_and_overlay_bubble(self, frame, face_coordinates, mistakes):
        for (x, y, w, h) in face_coordinates:
            scale_factor = 1 + (mistakes * 0.2)
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)

            # Ensure the bubble does not go beyond the frame dimensions
            new_width = min(new_width, frame.shape[1] - x)
            new_height = min(new_height, frame.shape[0] - y)

            resized_bubble = cv2.resize(self.bubble, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Ensure the resized bubble is correctly positioned
            x_offset = (new_width - w) // 2
            y_offset = (new_height - h) // 2

            # Adjust x, y to consider offsets (ensure x, y are not negative)
            x = max(x - x_offset, 0)
            y = max(y - y_offset, 0)

            # Calculate the end coordinates (ensure they do not go beyond the frame dimensions)
            x_end = min(x + new_width, frame.shape[1])
            y_end = min(y + new_height, frame.shape[0])

            # Recalculate the actual width and height to match the adjusted coordinates
            actual_width = x_end - x
            actual_height = y_end - y

            # Resize the bubble again if necessary
            if (new_width != actual_width) or (new_height != actual_height):
                resized_bubble = cv2.resize(self.bubble, (actual_width, actual_height), interpolation=cv2.INTER_AREA)

            # Create mask and inverse mask from the alpha channel
            mask = resized_bubble[:, :, 3]
            mask_inv = cv2.bitwise_not(mask)
            mask = cv2.merge([mask, mask, mask])
            mask_inv = cv2.merge([mask_inv, mask_inv, mask_inv])

            # Extract the ROI
            roi = frame[y:y_end, x:x_end]

            # Ensure mask is the correct type
            mask = mask.astype('uint8')
            mask_inv = mask_inv.astype('uint8')

            # Black-out the area of the bubble in ROI
            img1_bg = cv2.bitwise_and(roi, mask_inv)

            # Take only the region of the bubble from the bubble image
            img2_fg = cv2.bitwise_and(resized_bubble[:, :, :3], mask)

            # Put the bubble in ROI and modify the main image
            dst = cv2.add(img1_bg, img2_fg)
            frame[y:y_end, x:x_end] = dst

        return frame

    def overlay_bubble(self, frame, bubble, face_coordinates):
        x, y, w, h = face_coordinates
        y1, y2 = y, y + bubble.shape[0]
        x1, x2 = x, x + bubble.shape[1]

        alpha_s = bubble[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * bubble[:, :, c] +
                                      alpha_l * frame[y1:y2, x1:x2, c])
        return frame

