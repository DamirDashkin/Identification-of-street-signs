import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from ultralytics import YOLO
import numpy
#исходный код
if __name__ == '__main__':
  #выполнение кода до загрузки
  # initialise GUI
  top = tk.Tk()
  top.geometry('800x400')
  top.title('Traffic sign classification')
  top.configure(background='#1cb1ad')
  label = Label(top, background='#1cb1ad', font=('arial', 15, 'bold'))
  sign_image = Label(top)
  Valid_model = YOLO('D:/Diplom/Code/Diplom2024/yolov8n_416/weights/best.pt')
  def classify(file_path):
      global label_packed
      image = Image.open(file_path)
      results = Valid_model.predict([image], show=True, save = False,  imgsz=416, conf=0.5, iou=0.7)
      names = Valid_model.names
      print(names)
      result = ''
      sign = ''
      for r in results:
          for c in r.boxes.cls:
              if names[int(c)] == 'Speed Limit 10':
                  sign = 'Ограничение максимальной скорости 10 км\ч'
              elif names[int(c)] == 'Speed Limit 20':
                  sign = 'Ограничение максимальной скорости 20 км\ч'
              elif names[int(c)] == 'Speed Limit 30':
                  sign = 'Ограничение максимальной скорости 30 км\ч'
              elif names[int(c)] == 'Speed Limit 40':
                  sign = 'Ограничение максимальной скорости 40 км\ч'
              elif names[int(c)] == 'Speed Limit 50':
                  sign = 'Ограничение максимальной скорости 50 км\ч'
              elif names[int(c)] == 'Speed Limit 60':
                  sign = 'Ограничение максимальной скорости 60 км\ч'
              elif names[int(c)] == 'Speed Limit 70':
                  sign = 'Ограничение максимальной скорости 70 км\ч'
              elif names[int(c)] == 'Speed Limit 80':
                  sign = 'Ограничение максимальной скорости 80 км\ч'
              elif names[int(c)] == 'Speed Limit 90':
                  sign = 'Ограничение максимальной скорости 90 км\ч'
              elif names[int(c)] == 'Speed Limit 100':
                  sign = 'Ограничение максимальной скорости 100 км\ч'
              elif names[int(c)] == 'Speed Limit 110':
                  sign = 'Ограничение максимальной скорости 110 км\ч'
              elif names[int(c)] == 'Speed Limit 120':
                  sign = 'Ограничение максимальной скорости 120 км\ч'
              elif names[int(c)] == 'Stop':
                  sign = 'Стоп-знак'
              elif names[int(c)] == 'Green Light':
                  sign = 'Зеленый свет'
              elif names[int(c)] == 'Red Light':
                  sign = 'Красный свет'
              result = result + '\n' + sign
      print(result)
      label.configure(foreground = 'black', text = result)


  def classify_button(file_path):
      classify_b = Button(top, text="Распознать\nизображение", command=lambda: classify(file_path), padx=3, pady=40)

      classify_b.configure(background='#b11c5f', foreground='grey', font=('nyasha sans', 10, 'bold'))
      classify_b.place(relx=0.76, rely=0.34)

  def uploading_image():
      try:
          file_path = filedialog.askopenfilename()
          uploaded = Image.open(file_path)
          uploaded.thumbnail(((top.winfo_width() / 4.25), (top.winfo_height() / 4.25)))
          im = ImageTk.PhotoImage(uploaded)

          sign_image.configure(image=im)
          sign_image.image = im
          label.configure(text='Ваше изображение:')
          classify_button(file_path)
      except:
          pass


  upload = Button(top, text="Нажмите для загрузки изображения", command=uploading_image, padx=30, pady=10)
  upload.configure(background='#b11c5f', foreground='white', font=('nyasha sans', 10, 'bold'))
  upload.pack(side=BOTTOM, pady=50)
  sign_image.pack(side=BOTTOM, expand=True)
  label.pack(side=BOTTOM, expand=True)
  heading = Label(top, text="Узнайте Ваш дорожный знак", pady=20, font=('nyasha sans', 20, 'bold'))
  heading.configure(background='#1cb1ad', foreground='black')
  heading.pack()
  top.mainloop()
