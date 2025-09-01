import tkinter as tk
from tkinter import Canvas, Frame, Label, Button
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import cv2  

model = load_model("mnist_archive_combined.h5")

root = tk.Tk()
root.title("Handwritten Digit Recognizer")

canvas_width = 280
canvas_height = 280

frame = Frame(root)
frame.pack()

canvas = Canvas(frame, width=canvas_width, height=canvas_height, bg="white")
canvas.grid(row=0, column=0)

# draw faint gridlines
for i in range(0, canvas_width, 28):
    canvas.create_line([(i, 0), (i, canvas_height)], fill='lightgray')
for i in range(0, canvas_height, 28):
    canvas.create_line([(0, i), (canvas_width, i)], fill='lightgray')

output_label = Label(frame, text="Draw digits", font=("Helvetica", 24))
output_label.grid(row=0, column=1, padx=20)

image1 = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image1)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def clear():
    canvas.delete("all")
    for i in range(0, canvas_width, 28):
        canvas.create_line([(i, 0), (i, canvas_height)], fill='lightgray')
    for i in range(0, canvas_height, 28):
        canvas.create_line([(0, i), (canvas_width, i)], fill='lightgray')
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    output_label.config(text="Draw digits")

def predict():
    # convert PIL image to numpy
    img = np.array(image1)

    # invert (digits black on white)
    img = cv2.bitwise_not(img)

    # threshold to binary
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # find contours (digits)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # filter out tiny noise
            digit_img = thresh[y:y+h, x:x+w]
            # pad to square
            size = max(w, h)
            square = np.ones((size, size), dtype=np.uint8) * 0
            square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = digit_img

            # resize to 28x28
            digit_img = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
            digit_img = digit_img / 255.0
            digit_img = digit_img[np.newaxis, ..., np.newaxis]

            pred = model.predict(digit_img, verbose=0)
            digits.append((x, np.argmax(pred)))

    # sort left to right
    digits = sorted(digits, key=lambda x: x[0])
    number = "".join(str(d[1]) for d in digits)

    if number == "":
        output_label.config(text="No digits found")
    else:
        output_label.config(text=f"Predicted: {number}")

button_frame = Frame(root)
button_frame.pack(pady=10)

clear_button = Button(button_frame, text="Clear", command=clear)
clear_button.grid(row=0, column=0, padx=10)

predict_button = Button(button_frame, text="Predict", command=predict)
predict_button.grid(row=0, column=1, padx=10)

root.mainloop()

