import random
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import numpy as np


def pixelear():
    img_raw = cv.imread("imagen_global.jpg")
    roi = cv.selectROI(img_raw)
    roi_cropped = img_raw[int(roi[1]):int(roi[1] + roi[3]),
                  int(roi[0]):int(roi[0] + roi[2])]#recortar la imagen
    cv.imwrite("pixeleada.jpeg", roi_cropped)
    cv.waitKey(0)  # ventana de retención
    cv.destroyAllWindows()
    image = Image.open("pixeleada.jpeg")
    imgSmall = image.resize((16, 16), resample = Image.BILINEAR)
    resultado = imgSmall.resize(image.size,Image.NEAREST)
    resultado.save('pixel.jpeg')
    imgR= cv.imread('pixel.jpeg')
    ancho, alto = imgR.shape[:2]
    img_raw[roi[1]:(roi[1]+ancho), roi[0]:(roi[0]+alto)] = imgR[0:ancho,0:alto]
    image = imutils.resize(img_raw, height=600)
    # visuailizar la imagene en la interfas
    imagenToshow = imutils.resize(image, width=600)
    # MOSTRAR LA IMAGEN EN LA GUI
    img = Image.fromarray(imagenToshow)
    imgn = ImageTk.PhotoImage(image=img)
    ImgSalida.configure(image=imgn)
    ImgSalida.image = imgn

def zoomr():
    img_ra = cv.imread("imagen_global.jpg")  # leer la imagen
    roi = cv.selectROI(img_ra)  # funcion para seleccionar roi
    roi_cropped = img_ra[int(roi[1]):int(roi[1] + roi[3]),
                  int(roi[0]):int(roi[0] + roi[2])]  # Recortar roi seleccionado de la imagen sin procesar
    cv.imwrite("crop.jpeg", roi_cropped)
    cv.waitKey(0)  # ventana de retención
    cv.destroyAllWindows()
    imgR= cv.imread("crop.jpeg")
    image = imutils.resize(imgR, height=600)
    # visuailizar la imagene en la interfas
    imagenToshow = imutils.resize(image, width=600)
    # MOSTRAR LA IMAGEN EN LA GUI
    img = Image.fromarray(imagenToshow)
    imgn = ImageTk.PhotoImage(image=img)
    ImgSalida.configure(image=imgn)
    ImgSalida.image = imgn

def limpiar_panel():
    ImgSalida.image = ""
    selected.set(0)

def dezplazamiento():
	imagen_ax1 = Radiobutton(ventana,text="Dezplazamiento 1",width = 25, value = 1,variable=selec,command =desp)
	imagen_ax1.grid(column = 1, row =5)
	imagen_ax2 = Radiobutton(ventana,text="Dezplazamiento 2",width = 25, value = 2,variable=selec,command =desp )
	imagen_ax2.grid(column = 1, row =6)
	imagen_ax3 = Radiobutton(ventana,text="Dezplazamiento 3",width = 25, value = 3,variable=selec,command=desp)
	imagen_ax3.grid(column = 1, row =7)
	imagen_ax4 = Radiobutton(ventana,text="Dezplazamiento 4",width = 25, value = 4,variable=selec,command=desp)
	imagen_ax4.grid(column = 1, row =8)

def rotacion():
	imagen_ax1 = Radiobutton(ventana,text="0ª",width = 25, value = 1,variable=selec,command =esp)
	imagen_ax1.grid(column = 1, row =5)
	imagen_ax2 = Radiobutton(ventana,text="90ª",width = 25, value = 2,variable=selec,command =esp )
	imagen_ax2.grid(column = 1, row =6)
	imagen_ax3 = Radiobutton(ventana,text="180ª",width = 25, value = 3,variable=selec,command=esp)
	imagen_ax3.grid(column = 1, row =7)
	imagen_ax4 = Radiobutton(ventana,text="270ª",width = 25, value = 4,variable=selec,command=esp)
	imagen_ax4.grid(column = 1, row =8)

def espejo():
	imagen_ax1 = Radiobutton(ventana,text="Espejo 1",width = 25, value = 1,variable=selec,command =rt)
	imagen_ax1.grid(column = 1, row =5)
	imagen_ax2 = Radiobutton(ventana,text="espejo 2",width = 25, value = 2,variable=selec,command = rt)
	imagen_ax2.grid(column = 1, row =6)
	imagen_ax3 = Radiobutton(ventana,text="espejo 3",width = 25, value = 3,variable=selec,command=rt)
	imagen_ax3.grid(column = 1, row =7)
	imagen_ax4 = Radiobutton(ventana,text="espejo 4",width = 25, value = 4,variable=selec,command=rt)
	imagen_ax4.grid(column = 1, row =8)

def desp():
    limpiar_panel()
    lblInfor = Label(ventana, text = " IMAGEN SELECCIONADA")
    lblInfor.grid(column=2,row=1,columnspan=2)
    imagen = cv.imread("imagen_global.jpg")
    alto, ancho = imagen.shape[:2]
    limpiar_panel()
    img4 = cv.imread("imagen_global.jpg")
    if selec.get() == 1:
        # Traslacion
        # 1
        M = np.float32([[1, 0, 0], [0, 1, -10]])
        imagenSalida = cv.warpAffine(img4, M, (alto, ancho))
        image = imutils.resize(imagenSalida, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
    if selec.get() == 2:
        # Traslacion
        # 2
        M = np.float32([[1, 0, -10], [0, 1, 0]])
        imagenSalida2 = cv.warpAffine(img4, M, (alto, ancho))
        image = imutils.resize(imagenSalida2, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
        # Traslacion
        # 3
    if selec.get() == 3:

        M = np.float32([[1, 0, 0], [0, 1, 10]])
        imagenSalida3 = cv.warpAffine(img4, M, (alto, ancho))
        image = imutils.resize(imagenSalida3, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
        # Traslacion

    if selec.get() == 4:
        # 4
        M = np.float32([[1, 0, 10], [0, 1, 0]])
        imagenSalida4 = cv.warpAffine(img4, M, (alto, ancho))
        image = imutils.resize(imagenSalida4, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

def esp():
    limpiar_panel()
    lblInfor = Label(ventana, text = " IMAGEN SELECCIONADA")
    lblInfor.grid(column=2,row=1,columnspan=2)
    imagen = cv.imread("imagen_global.jpg")
    img5 = imagen
    alto, ancho = imagen.shape[:2]

    if selec.get() == 1:
        image = imutils.resize(img5, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
    if selec.get() == 4:
        # 1
        Mt = cv.getRotationMatrix2D((alto // 2, ancho // 2), 90, 1)
        imgRt = cv.warpAffine(img5, Mt, (alto, ancho))
        image = imutils.resize(imgRt, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

    if selec.get() == 3:
        # 2
        Mt2 = cv.getRotationMatrix2D((alto // 2, ancho // 2), 180, 1)
        imgRt1 = cv.warpAffine(img5, Mt2, (ancho, alto))
        image = imutils.resize(imgRt1, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
    if selec.get() == 2:
        # 3
        Mt3 = cv.getRotationMatrix2D((alto // 2, ancho // 2), 270, 1)
        imgRt2 = cv.warpAffine(img5, Mt3, (alto, ancho))
        image = imutils.resize(imgRt2, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

def rt():
    limpiar_panel()
    lblInfor = Label(ventana, text=" IMAGEN SELECCIONADA")
    lblInfor.grid(column=2, row=1, columnspan=2)
    imagen = cv.imread("imagen_global.jpg")
    alto, ancho = imagen.shape[:2]
    img6 = imagen
    imgax = imagen
    if selec.get() == 1:
        image = imutils.resize(img6, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

    if selec.get() == 2:
        imagen = cv.imread("imagen_global.jpg")
        img6 = imagen
        contAn = ancho - 1
        for i in range(0, alto):
            for j in range(0, ancho):
                img6[i][j] = imgax[i][contAn]
                contAn = contAn - 1
            contAn = ancho - 1

        image = imutils.resize(img6, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
    if selec.get() == 3:
        # muestra la imagen cambiada
        # crear la imagen horizontal
        contAl = alto - 1
        imagen = cv.imread("imagen_global.jpg")
        img6 = imagen
        for i in range(0, alto):
            for j in range(0, ancho):
                img6[i][j] = imgax[contAl][j]
            contAl = contAl - 1

        image = imutils.resize(img6, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
    if selec.get() == 4:
        # imagen respecto a horizontal y vertical
        contAn = ancho - 1
        contAl = alto - 1
        imagen = cv.imread("imagen_global.jpg")
        img6 = imagen
        for i in range(0, alto):
            for j in range(0, ancho):
                img6[i][j] = imgax[contAl][contAn]
                contAn = contAn - 1
            contAl = contAl - 1
            contAn = ancho - 1
        image = imutils.resize(img6, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

def histograma():
    imagen = cv.imread("imagen_global.jpg")
    plt.hist(imagen.ravel(), 256, [0, 256])  # rango de 0 a 256
    plt.show()  # muestra el histograma

def ec_histo():
    img = cv.imread("imagen_global.jpg", cv.IMREAD_GRAYSCALE)
    ancho, alto = img.shape
    x = np.linspace(0, 255, num=256, dtype=np.uint8)
    y = np.zeros(256)
    y_aux = np.zeros(256)

    for i in range(ancho):
        for h in range(alto):
            v = img[i, h]
            y[v] = y[v] + 1

    total = 255 / (ancho * alto)
    suma = 0
    nueva = np.zeros(img.shape, img.dtype)
    for i in range(ancho):
        for j in range(alto):
            for s in range(img[i, j]):
                suma = suma + y[s]
            nueva[i, j] = total * suma
            suma = 0
    for i in range(ancho):
        for h in range(alto):
            v = nueva[i, h]
            y_aux[v] = y_aux[v] + 1

    cv.imwrite('ecualizacion.jpg', nueva)
    ecu = cv.imread('ecualizacion.jpg')
    image = imutils.resize(ecu, height=600)
    # visuailizar la imagene en la interfas
    imagenToshow = imutils.resize(image, width=600)
    # MOSTRAR LA IMAGEN EN LA GUI
    img = Image.fromarray(imagenToshow)
    imgn = ImageTk.PhotoImage(image=img)
    ImgSalida.configure(image=imgn)
    ImgSalida.image = imgn
    plt.subplot(1,2,1),plt.bar(x,y)
    plt.subplot(1,2,2),plt.bar(x,y_aux)
    plt.show()

def aleatoria():
    img_aleatoria = []
    img_aleatoriaax = []
    for i in range(600):
        img_aleatoria.append([])
        for j in range(600):
            img_aleatoria[i].append(random.randint(0, 255))

    escribir = open('imagen_ale.pgm', 'w')
    escribir.write("P2\n")
    escribir.write("600 600\n")
    escribir.write("255\n")
    for h in range(len(img_aleatoria)):
        escribir.write(str(img_aleatoria[h]))
        escribir.write("\n")
    escribir.close()

    imagen = cv.imread("ImagenAleatoria.pgm")
    cv.imshow("Binaria", imagen)
    cv.waitKey(0)
    cv.destroyAllWindows()

def parte2():
    lblInfo = Label(ventana, text = " IMAGEN SELECCIONADA")
    lblInfo.grid(column=2,row=1,columnspan=2)
    imagen=cv.imread("imagen_global.jpg")
    alto, ancho = imagen.shape[:2]
    if selected.get() ==2:
        limpiar_panel()
        eliminar_column()
        limpieza =cv.imread("imagen_global.jpg")
        limpieza[150:250, 100:600] = 0
        image = imutils.resize(limpieza, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

    if selected.get()==3:
        limpiar_panel()
        eliminar_column()

        imagen=cv.imread("imagen_global.jpg")
        negativo = np.zeros(imagen.shape, imagen.dtype)
        for i in range(alto):
            for j in range(ancho):
                negativo[i][j] = 255 - imagen[i, j]
        image = imutils.resize(negativo, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
        negativo = ""

    if selected.get() == 7:
        limpiar_panel()
        eliminar_column()
        imagen = cv.imread("imagen_global.jpg")
        img7=imagen
       # loga = np.zeros(img7.shape, img7.dtype)
        c = 1
        loga = c * np.log(1 + img7)
        maxi = np.amax(loga)
        loga = np.uint8(loga / maxi * 255)
        image = imutils.resize(loga, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

    if selected.get() == 8:
        limpiar_panel()
        eliminar_column()
        imagen = cv.imread("imagen_global.jpg")
        img8=imagen
        pot = np.zeros(img8.shape, img8.dtype)
        c = 1
        e=np.exp(1)
        pot = c * np.power(img8,e)
        maxi = np.amax(pot)
        exp = np.uint8(pot/maxi*255)
        image = imutils.resize(exp, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

    if selected.get() == 9:
        limpiar_panel()
        eliminar_column()
        imagen = cv.imread("imagen_global.jpg")
        img9=imagen
        gamma= .6
        lookupTable= np.empty((1,256),np.uint8)
        for i in range (256):
            lookupTable[0,i]= np.clip(pow(i / 255.0, gamma)*255.0,0,255)

        resultado = cv.LUT(img9,lookupTable)
        image = imutils.resize(resultado, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

    if selected.get() == 10:
        limpiar_panel()
        eliminar_column()
        imagen = cv.imread("imagen_global.jpg")
        img10=imagen
        beta= 75
        gamma=120
        imgAuxi=np.int16(img10)
        imgAuxi=imgAuxi * (gamma/127+1)-gamma+beta
        imgAuxi=np.clip(imgAuxi,0,255)
        imgAuxi=np.uint8(imgAuxi)
        image = imutils.resize(imgAuxi, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn
    if selected.get() == 11:
        limpiar_panel()
        eliminar_column()
        imagen = cv.imread("imagen_global.jpg")
        imagenCop=imagen
        imgColor=imagen
        alto, ancho = imgColor.shape[:2]
        imagenCop[0:200,0:200]=imgColor[200:400,200:400]
        image = imutils.resize(imagenCop, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

    if selected.get() == 13:
        limpiar_panel()
        eliminar_column()
        imagen = cv.imread("imagen_global.jpg")
        image = imutils.resize(imagen, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
          # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

def abrirImagen():
    ImagenFiltrada = filedialog.askopenfilename(filetypes=[
        ("todos los fomratos de imagen", ".jpg"),
        ("todos los fomratos de imagen", ".jpg"),
        ("todos los fomratos de imagen", ".jpg")])
    if (len(ImagenFiltrada) > 0):
        # leer la imagen de entrada
        imagen = cv.imread(ImagenFiltrada, cv.IMREAD_GRAYSCALE)
        cv.imwrite("imagen_global.jpg",imagen)

        image = imutils.resize(imagen, height = 380)
        #visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=180)
       # MOSTRAR LA IMAGEN EN LA GUI

        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image = img)

        lblInputImagen.configure(image = imgn)
        lblInputImagen.image = imgn
        imagen = cv.imread("imagen_global.jpg")
        image = imutils.resize(imagen, height=600)
        # visuailizar la imagene en la interfas
        imagenToshow = imutils.resize(image, width=600)
        # MOSTRAR LA IMAGEN EN LA GUI
        img = Image.fromarray(imagenToshow)
        imgn = ImageTk.PhotoImage(image=img)
        ImgSalida.configure(image=imgn)
        ImgSalida.image = imgn

def eliminar_column():
    imagen_ax1 = Label(ventana, text=" ", width=25)
    imagen_ax1.grid(column=1, row=5)
    imagen_ax2 = Label(ventana, text=" ", width=25,)
    imagen_ax2.grid(column=1, row=6)
    imagen_ax3 = Label(ventana, text=" ", width=25)
    imagen_ax3.grid(column=1, row=7)
    imagen_ax4 =Label(ventana, text=" ", width=25)
    imagen_ax4.grid(column=1, row=8)

def promedio():
    limpiar_panel()
    eliminar_column()
    imagen = cv.imread("imagen_global.jpg")
    f1 = cv.blur(imagen,(3,3))
    image = imutils.resize(f1, height=600)
    # visuailizar la imagene en la interfas
    imagenToshow = imutils.resize(image, width=600)
    # MOSTRAR LA IMAGEN EN LA GUI
    img = Image.fromarray(imagenToshow)
    imgn = ImageTk.PhotoImage(image=img)
    ImgSalida.configure(image=imgn)
    ImgSalida.image = imgn

def gauss():
    limpiar_panel()
    eliminar_column()
    imagen = cv.imread("imagen_global.jpg")
    f2 = cv.GaussianBlur(imagen,(3,3),0)
    image = imutils.resize(f2, height=600)
    # visuailizar la imagene en la interfas
    imagenToshow = imutils.resize(image, width=600)
    # MOSTRAR LA IMAGEN EN LA GUI
    img = Image.fromarray(imagenToshow)
    imgn = ImageTk.PhotoImage(image=img)
    ImgSalida.configure(image=imgn)
    ImgSalida.image = imgn

def mediana():
    limpiar_panel()
    eliminar_column()
    imagen = cv.imread("imagen_global.jpg")
    f3 = cv.medianBlur(imagen,5)
    image = imutils.resize(imagen, height=600)
    # visuailizar la imagene en la interfas
    imagenToshow = imutils.resize(image, width=600)
    # MOSTRAR LA IMAGEN EN LA GUI
    img = Image.fromarray(imagenToshow)
    imgn = ImageTk.PhotoImage(image=img)
    ImgSalida.configure(image=imgn)
    ImgSalida.image = imgn
def convolucion():
    limpiar_panel()
    eliminar_column()
    imagen = cv.imread("imagen_global.jpg")
    #creando un kernel
    f4 = np.ones((3,3),np.float32)/25
    f4f = cv.filter2D(imagen,-1,f4)
    image = imutils.resize(f4f, height=600)
    # visuailizar la imagene en la interfas
    imagenToshow = imutils.resize(image, width=600)
    # MOSTRAR LA IMAGEN EN LA GUI
    img = Image.fromarray(imagenToshow)
    imgn = ImageTk.PhotoImage(image=img)
    ImgSalida.configure(image=imgn)
    ImgSalida.image = imgn

def salir():

    exit()

ventana = Tk()
#CREACION DE LABELS
ventana.title("Procesamiento de imagenes digitales")

lblselec = Label(ventana, text=" IMAGEN SELECCIONADA")
lblselec.grid(column=0, row=0)
eliminar_column()
lblInfo = Label(ventana, text=" VISUALISAR IMAGEN")
lblInfo.grid(column=2, row=1, columnspan=2)
lblInputImagen = Label(ventana)
lblInputImagen.grid(column = 0, row = 2)
ImgSalida = Label(ventana)
ImgSalida.grid(column=2, row=2, rowspan=14, columnspan=2)
labelsc = Label(ventana,text= "OPERACIONES EN ESCALA DE GRISES",width= 30)
labelsc.grid(column = 0, row=4)
# crear un menu para operaciones auxiliares

menus = Menu(ventana)
menu_histograma = Menu(menus)
imal = Menu(menus)
abrir = Menu(menus)
filtros = Menu(menus)
filtros.add_command(label='Promedio',command=promedio)
filtros.add_command(label='Gaussiano',command=gauss)
filtros.add_command(label='Mediana',command=mediana)
filtros.add_command(label='Convolucion',command=convolucion)
menu_histograma.add_command(label="histograma", command=histograma)
menu_histograma.add_command(label='Ecualizacion del histo', command = ec_histo)
abrir.add_command(label='Abrir',command = abrirImagen)
abrir.add_command(label='Salir',command = salir)
imal.add_command(label = 'Imagen aleatoria',command=aleatoria)
menus.add_cascade(label='Archivo', menu=abrir)
menus.add_cascade(label='Histogramas', menu=menu_histograma)
menus.add_cascade(label='Img aleatoria', menu=imal)
menus.add_cascade(label='Filtros', menu=filtros)

ventana.config(menu=menus)
#CREACION DE BOTONES

zoom = Button(ventana,text="ZOOM CON ROI",width=25,command = zoomr)
zoom.grid(column=4,row=5,padx=5,pady=5,columnspan=2)
zoom = Button(ventana,text="PIX CON ROI",width=25,command = pixelear)
zoom.grid(column=4,row=6,padx=5,pady=5,columnspan=2)

#CREACION DE RADIOSBUTTON
selec = IntVar()
selected = IntVar()

radiobtn2=Radiobutton(ventana,text ="Limpieza           ",width= 25 , value = 2 ,variable = selected,command= parte2)
radiobtn2.grid(column=0,row = 6,padx=5)
radiobtn3=Radiobutton(ventana,text ="Inversion(negativo)",width= 25 , value = 3 ,variable = selected,command= parte2)
radiobtn3.grid(column=0,row = 7,padx=5)
radiobtn4=Radiobutton(ventana,text ="Desplazamiento     ",width= 25 , value = 4 ,variable = selected,command= dezplazamiento)
radiobtn4.grid(column=0,row = 8,padx=5)
radiobtn5=Radiobutton(ventana,text ="Rotacion           ",width= 25 , value = 5 ,variable = selected,command= rotacion)
radiobtn5.grid(column=0,row = 9,padx=5)
radiobtn6=Radiobutton(ventana,text ="Espejo             ",width= 25 , value = 6 ,variable = selected,command= espejo)
radiobtn6.grid(column=0,row = 10,padx=5)
radiobtn7=Radiobutton(ventana,text ="Log de una imagen  ",width= 25 , value = 7 ,variable = selected,command= parte2)
radiobtn7.grid(column=0,row = 11,padx=5)
radiobtn8=Radiobutton(ventana,text ="Potencia           ",width= 25 , value = 8 ,variable = selected,command= parte2)
radiobtn8.grid(column=0,row = 12,padx=5)
radiobtn9=Radiobutton(ventana,text ="Brillo             ",width= 25 , value = 9 ,variable = selected,command= parte2)
radiobtn9.grid(column=0,row = 13,padx=5)
radiobtn10=Radiobutton(ventana,text ="Contraste         ",width= 25 , value =10 ,variable = selected,command= parte2)
radiobtn10.grid(column=0,row = 14,padx=5)
radiobtn11=Radiobutton(ventana,text ="Copia             ",width= 25 , value =11 ,variable = selected,command= parte2)
radiobtn11.grid(column=0,row = 15,padx=5)
radiobtn12=Radiobutton(ventana,text ="Imagen Original   ",width= 25 , value =13 ,variable = selected,command= parte2)
radiobtn12.grid(column=0,row = 5,padx=5)


ventana.mainloop()