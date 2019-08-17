#!/usr/bin/env python

#Dependencias necesarias
import os
import cv2
import imagehash
import numpy as np

from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

#Directorio donde se encuentran las imagenes originales
OriginDirectory = 'OriginalImages'

#Directorio donde se guardaran las imagenes filtradas y redimensionadas
DestinationDirectory = 'FilterImages'

#Estension para guardar las imagenes
ExtensionToSave = '.jpg'

def reSize(img, type):
	#Escalar imagen de tamaño fijo 256x256px
	if(type=='fixed'):
		return cv2.resize(img,(256,256))

	#Escalar imagen a un tamaño proporcional
	elif(type=='proportional'):
		W = 2000
		height, width, depth = img.shape
		imgScale = W/width
		newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
		return cv2.resize(img,(int(newX),int(newY)))

def Save(img, name, ext, dest):
	cv2.imwrite(dest + '/' + name + ext, img)

def crop(img,xpoint,ypoint,dist):
	y1 = ypoint - dist
	y2 = ypoint + dist
	x1 = xpoint - dist
	x2 = xpoint + dist
	crop = img[y1:y2, x1:x2]
	return crop

def Clustering(arr, tot):
	X = np.array(arr)

	kmeans = KMeans(n_clusters=tot).fit(X)
	centroids = kmeans.cluster_centers_

	return centroids

def SearchCircle(img):

	#Imagen base redimensionada proporcionalmente a la original
	baseimg = reSize(img, 'proportional')

	#Imagen igual a la base, para aplicar circulos mas adelante
	circleimg = reSize(img, 'proportional')

	#Imagen igual a la base, para aplicar el resultado filtrado
	filterimg = reSize(img, 'proportional')

	#Imagen con filtros
	grayimg = cv2.cvtColor(baseimg, cv2.COLOR_BGR2GRAY)
	gaussianimg = cv2.GaussianBlur(grayimg, (5,5), 0)

	#Parametros para busqueda de circulos
	param1 = 50
	param2 = 30
	minRadius = 700
	maxRadius = 1000

	centers = list()
	rads = list()

	circles = cv2.HoughCircles(gaussianimg,cv2.HOUGH_GRADIENT,1,20, param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)

	#codigo para graficar circulos
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:

			xvar = i[0]
			yvar = i[1]

			center = (xvar,yvar)
			centers.append(center)

			radius = i[2]
			rads.append(radius)

			stroke = 5

			#Circulo
			cv2.circle(circleimg,center,radius,(0,0,255),stroke)
			#Centro
			cv2.circle(circleimg,center,5,(0,255,0),stroke)
		
		centroid = Clustering(centers, 1)
		centroid = np.uint16(np.around(centroid))

		cxvar = centroid[0][0]
		cyvar = centroid[0][1]
		clusterCenter = (cxvar,cyvar)
		maxItem = max(rads)

		#Circulo
		cv2.circle(filterimg,clusterCenter,maxItem,(255,0,0),stroke)
		#centro
		cv2.circle(filterimg,clusterCenter,5,(255,0,0),stroke)

		cropimg = crop(baseimg,cxvar,cyvar,maxItem)

	else:
		print('No hay circulos')


	#**********Graficar imagenes**********
	fig = plt.figure(1)

	ax1 = fig.add_subplot(231)
	ax2 = fig.add_subplot(232)
	ax3 = fig.add_subplot(233)
	ax4 = fig.add_subplot(234)
	ax5 = fig.add_subplot(235)
	ax6 = fig.add_subplot(236)

	ax1.title.set_text('Original')
	ax1.imshow(baseimg)

	ax2.title.set_text('escala de grises / desenfoque gaussiano')
	ax2.imshow(gaussianimg, cmap=plt.cm.gray)

	ax3.title.set_text('perimetros / centros')
	ax3.imshow(circleimg)

	ax4.title.set_text('perimetro mayor / centroide')
	ax4.imshow(filterimg)

	ax5.title.set_text('imagen centrada y recortada')
	ax5.imshow(cropimg)

	ax6.title.set_text('redimensionada')

	plt.show()
	#****************************************

	#retornar la misma imagen que se recibe, solo para no bloquear el proceso principal al modificar esta definicion
	#return img

def BruteForce(Orgn):
	for x in range(0, len(Images)-1):
		Img1 = cv2.imread(Orgn + '/' + Images[x])
		for y in range(x+1, len(Images)):
			Img2 = cv2.imread(Orgn + '/' + Images[y])


			if Img1.shape == Img2.shape:
				text = 'mismos canales y tamaño'
				difference = cv2.subtract(Img1, Img2)
				b,g,r = cv2.split(difference)

				if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
					text = 'iguales'
			else:
				text = 'distintas'

			print(str(x) + ", " + str(y) + "; " + str(Images[x] + ", " + Images[y] + "; " + text))

try:
	Images = os.listdir(OriginDirectory)
	print('Total de imagenes: ' + str(len(Images)))

	for x in range(0, len(Images)):
		print(Images[x])
		CircleImg = SearchCircle(cv2.imread(OriginDirectory + '/' + Images[x]))

	
	#for x in range(0, len(Images)):
		#ImgRoute = OriginDirectory + '/' + Images[x]
		#hash = imagehash.average_hash(Image.open(ImgRoute))


		#CircleImg = SearchCircle(cv2.imread(ImgRoute))
		#ReSizeImg = reSize(CircleImg, 'fixed')

		#Falta codigo para seleccionar la mejor imagen
		#Save(ReSizeImg, str(hash), ExtensionToSave, DestinationDirectory)
		
		#Save(cv2.imread(ImgRoute), str(hash), ExtensionToSave, DestinationDirectory)
		#print(hash)
	

except KeyboardInterrupt:
	exit()
#except:
#	print("Error")


#Fuzzy hashing
#Radix Sort, Quick Sort