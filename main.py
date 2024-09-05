import tkinter as tk
import random
from PIL import Image, ImageTk
import cv2
from FaceDetector import FaceDetector
from BubbleOverlay import BubbleOverlay
from GameControl import GameControl
from WinGame import WinGame

class HangmanGame:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Hangman")

        self.words_list = ["python", "hangman", "computer", "game"]
        self.selected_word = random.choice(self.words_list)
        self.guessed_letters = []
        self.max_mistakes = 6
        self.mistakes_made = 0
        self.display_word = ["_" for _ in self.selected_word]

        self.bubble_overlay = BubbleOverlay('bubble.png')
        self.face_detector = FaceDetector()

        self.setup_gui()
        self.update_display_word()

        self.cap = cv2.VideoCapture(0)
        self.update_video()

        # Game Over
        self.game_control = GameControl("GameOver.jpg")

        # Game Won
        self.game_won = WinGame("Crown.png", "Firework.png")

    def setup_gui(self):
        self.word_label = tk.Label(self.window, text=" ".join(self.display_word), font=("Helvetica", 24))
        self.word_label.pack(pady=20)

        self.buttons_frame = tk.Frame(self.window)
        self.buttons_frame.pack(pady=20)

        for char in 'abcdefghijklmnopqrstuvwxyz':
            btn = tk.Button(self.buttons_frame, text=char, command=lambda c=char: self.guess_letter(c), width=4,
                            height=2)
            btn.pack(side='left')
        self.mistakes_label = tk.Label(self.window, text=f"Mistakes: {self.mistakes_made}/{self.max_mistakes}",
                                       font=("Helvetica", 16))
        self.mistakes_label.pack()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            faces = self.face_detector.detect_faces(frame)

            # If faces are detected Overlay the bubble on the frame
            if faces is not None and len(faces) > 0:
                frame = self.bubble_overlay.resize_and_overlay_bubble(frame, faces, self.mistakes_made)

            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
            imgtk = ImageTk.PhotoImage(image=img)

            if not hasattr(self, 'video_label'):
                self.video_label = tk.Label(self.window, image=imgtk)
                self.video_label.imgtk = imgtk
                self.video_label.pack(side="left")
            else:
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

            self.window.after(10, self.update_video)


    def guess_letter(self, letter):
        if letter in self.selected_word and letter not in self.guessed_letters:
            self.guessed_letters.append(letter)
            self.update_display_word()
        elif letter not in self.selected_word:
            self.mistakes_made += 1
            self.mistakes_label.config(text=f"Mistakes: {self.mistakes_made}/{self.max_mistakes}")
            if self.mistakes_made >= self.max_mistakes:
                self.end_game(False)

        if "_" not in self.display_word:
            self.end_game(True)

    def update_display_word(self):
        for i, char in enumerate(self.selected_word):
            if char in self.guessed_letters:
                self.display_word[i] = char
        self.word_label.config(text=" ".join(self.display_word))

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def end_game(self, won):
        for widget in self.buttons_frame.winfo_children():
            widget.config(state="disabled")

        win_lose_text = "Congratulations, you won!" if won else f"You lost! The word was: {self.selected_word}"
        tk.Label(self.window, text=win_lose_text, font=("Helvetica", 20)).pack()

        if won:
            self.window.destroy()
            cv2.waitKey(500)
            self.game_won.transition_to_win_screen(self.cap, self.face_detector)
            self.game_won.end_game()

        else:
            self.window.destroy()
            cv2.waitKey(500)

            self.game_control.transition_to_game_over(self.cap, self.selected_word)
            self.game_control.end_game()
            self.cap.release()




if __name__ == "__main__":
    game = HangmanGame()
    game.window.mainloop()



