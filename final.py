import cv2, pickle
import numpy as np
# import tensorflow as tf
#from cnn_tf import cnn_model_fn
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
import pytesseract
from PIL import Image
import time

welcome = pyttsx3.init()
welcome.say("Welcome to HSBC Interpreter. How can I help you?")
welcome.runAndWait()

engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.h5')

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def get_image_size():
	img = cv2.imread('gestures/1/1.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab*100 > 20:
		text = get_pred_text_from_db(pred_class)
	return text

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300
is_voice_on = True

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

def say_text(text):
	if not is_voice_on:
		return
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

def red_transfer():

	file1 = open('sample1.txt', 'r')
	Lines = file1.readlines()
	data =""
	for line in Lines:
		if line[0] == "S":
			data = data + "Debit Account is " + line
		elif line[0] == "V":
			data = data + "Credit Account is " + line
	return data

def red_text():

	file1 = open('sample.txt', 'r')
	Lines = file1.readlines()
	data =""
	for line in Lines:
		if line[0] == "G" or line[0] == "S" or line[0] == "1":
			if line[0:3] == "106":
				data = data + "Account Number" + line;
			elif line[0:3] == "10.":
				data = data + "Balance is " + line;
			else:
				data = data + line

	return data

def text_mode(cam):
	global is_voice_on
	flag = 1
	text = ""
	word = ""
	word1= ""
	count_same_frame = 0
	logo = cv2.imread('HSBC.JPG')
	logo = cv2.resize(logo, (640, 480))
	logo_dim = np.array(logo)
	hsbc = cv2.imread("D:/Voyager/Sign-Language-Interpreter-using-Deep-Learning-master/Sign-Language-Interpreter-using-Deep-Learning-master/Code/View/Account.jpg")
	Transfer = cv2.imread("D:/Voyager/Sign-Language-Interpreter-using-Deep-Learning-master/Sign-Language-Interpreter-using-Deep-Learning-master/Code/View/Transfer.jpg")
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0

				if count_same_frame > 10:
					if len(text) == 1:
						Thread(target=say_text, args=(text, )).start()
					word = word + text
					if word.startswith('I/Me '):
						word = word.replace('I/Me ', 'I ')
					elif word.endswith('I/Me '):
						word = word.replace('I/Me ', 'me ')
					elif word == "goodthank you":
						word = "I am good. Thank you!"
					elif word == "askhelp":
						word = "I need to ask for some help"
					elif word == "credit carddue date":
						word = "I need help with my credit card due"
					elif word == "extend":
						word = "Can you extend my due date?"
					elif word == "emergencycovid 19":
						word = " I have medical emergency regarding covid 19"
					elif word == "thank you":
						word = "Thank you very much!"
					elif word == "yes":
						word = "Yes, please tell me about it"
					elif word == "email details":
						word = "Please email me the details"
					elif word == "thank younice":
						word = "Thank you! Have a nice day"


					# elif word == "transfer":
					# 	hsbc = cv2.imread(
					# 		"D:/Voyager/Sign-Language-Interpreter-using-Deep-Learning-master/Sign-Language-Interpreter-using-Deep-Learning-master/Code/View/Transfer.jpg")
					# 	word = red_transfer()
					# elif word == "make transfer":
					# 	hsbc = cv2.imread(
					# 		"D:/Voyager/Sign-Language-Interpreter-using-Deep-Learning-master/Sign-Language-Interpreter-using-Deep-Learning-master/Code/View/done.jpg")
					# 	word = "Transaction completed"
					count_same_frame = 0

			elif cv2.contourArea(contour) < 1000:
				if word != '':
					#print('yolo')
					#say_text(text)
					Thread(target=say_text, args=(word, )).start()
					# Thread(target=say_text, args=(word1,)).start()
				text = ""
				word = ""
				word1 = ""
		else:
			if word != '':
				#print('yolo1')
				#say_text(text)
				Thread(target=say_text, args=(word, )).start()
				# Thread(target=say_text, args=(word1,)).start()
			text = ""
			word = ""
			word1 = ""


		whiteboard = np.ones((480, 640, 3), dtype=np.uint8)
		whiteboard = np.multiply(whiteboard,logo_dim)
		cv2.putText(whiteboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
		cv2.putText(whiteboard, "Predicted text- " + text, (30, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 0))
		cv2.putText(whiteboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
		if word1!="":
			cv2.putText(whiteboard, "Manager :" + word1, (30, 360), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
		if is_voice_on:
			cv2.putText(whiteboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		else:
			cv2.putText(whiteboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, whiteboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)

		# cv2.imshow("HSBC", hsbc)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('c'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True

	if keypress == ord('c'):
		return 2
	else:
		return 0

def recognize():
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		elif keypress == 2:
			keypress = calculator_mode(cam)
		else:
			break

# value = Image.open("D:/Voyager/Sign-Language-Interpreter-using-Deep-Learning-master/Sign-Language-Interpreter-using-Deep-Learning-master/Code/View/Account.jpg")
#
# pytesseract.pytesseract.tesseract_cmd = r'C:/Users/prana/AppData/Local/Tesseract-OCR/tesseract.exe'
# data = pytesseract.image_to_string(value, config='--tessdata-dir "C:/Users/prana/AppData/Local/Tesseract-OCR/tessdata"')
#
# text_file = open("sample.txt", "w")
# text_file.writelines(data)
# text_file.close()
#
#
# value_transfer = Image.open("D:/Voyager/Sign-Language-Interpreter-using-Deep-Learning-master/Sign-Language-Interpreter-using-Deep-Learning-master/Code/View/Transfer.jpg")
#
# # text = pytesseract.image_to_string(value, config='--tessdata-dir "C://Tesseract-OCR//tessdata"')
#
# pytesseract.pytesseract.tesseract_cmd = r'C:/Users/prana/AppData/Local/Tesseract-OCR/tesseract.exe'
# data_transfer = pytesseract.image_to_string(value_transfer, config='--tessdata-dir "C:/Users/prana/AppData/Local/Tesseract-OCR/tessdata"')
#
# transfer_file = open("sample1.txt", "w")
# transfer_file.writelines(data_transfer)
# transfer_file.close()

keras_predict(model, np.zeros((50, 50), dtype = np.uint8))		
recognize()
