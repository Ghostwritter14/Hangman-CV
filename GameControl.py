import cv2


class GameControl:
    def __init__(self, image_path, ):
        self.game_over_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.cap = cv2.VideoCapture(0)
        if self.game_over_image is None:
            raise FileNotFoundError(f"Game over image not found at {image_path}")

    def transition_to_game_over(self, cap, word):
        steps = 30
        for i in range(steps):
            ret, frame = cap.read()
            if not ret:
                break

            alpha = i / steps
            game_over_resized = cv2.resize(self.game_over_image, (frame.shape[1], frame.shape[0]))
            blended_frame = cv2.addWeighted(frame, 1 - alpha, game_over_resized, alpha, 0)
            cv2.imshow('Game Over', blended_frame)
            cv2.waitKey(int(1000 / steps))

        # show final message
        self.show_final_message(game_over_resized, word)

    def show_final_message(self, final_frame, word):
        text = f"You lost! The word was: {word}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        textX = (final_frame.shape[1] - textsize[0]) // 2
        textY = (final_frame.shape[0] + textsize[1]) // 2

        cv2.putText(final_frame, text, (textX, textY), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Game Over', final_frame)
        cv2.waitKey(0)

    def end_game(self):
        cv2.destroyAllWindows()
        self.cap.release()
