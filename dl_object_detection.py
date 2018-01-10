# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the packages
import numpy as np
import argparse
import cv2

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

def detect(bot, update):
	print('run detect function')

	#save input image as input.jpg
	file_id = update.message.photo[-1].file_id
	newFile = bot.get_file(file_id)
	newFile.download('Input.jpg')
	print('File saved as Input.jpg')
	
	print('Start to detect')
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	'''ap.add_argument('-i', '--image', required=True,
		help='path to input image')
	ap.add_argument('-p', '--prototxt', required=True,
		help='path to Caffe 'deploy' prototxt file')
	ap.add_argument('-m', '--model', required=True,
		help='path to Caffe pre-trained model')'''
	ap.add_argument('-c', '--confidence', type=float, default=0.2,
		help='minimum probability to filter weak detections')
	args = vars(ap.parse_args())

	'''--prototxt MobileNetSSD_deploy.prototxt.txt \
				--model MobileNetSSD_deploy.caffemodel \
				--image images/example_06.jpg'''

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class

	CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
		'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
		'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
		'sofa', 'train', 'tvmonitor']
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	print('[INFO] loading model...')
	#net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
	net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel') 

	# load the input image and construct an input blob for the imagex
	# by resizing to a fixed 300x300 pixels and then normalizing it
	# (note: normalization is done via the authors of the MobileNet SSD
	# implementation)
	###image = cv2.imread(args['image'])
	image = cv2.imread('Input.jpg')
	print('imread OK')

	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	print('[INFO] computing object detections...')
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		print(i, 'in ', detections.shape[2])
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args['confidence']:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype('int')

			# display the prediction
			label = '{}: {:.2f}%'.format(CLASSES[idx], confidence * 100)
			print('[INFO] {}'.format(label))
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output image
	#cv2.imshow('Output', image)

	#save the output image
	cv2.imwrite('Output.jpg', image)
	#cv2.waitKey(0)
	
	#send Output (detected) photo
	bot.send_photo(update.message.chat_id, photo = open('Output.jpg', 'rb'))

def file_error(bot, update):
	print('Please send me the picture as a \'Photo\', not as a \'File\'.')
	bot.sendMessage(update.message.chat_id, text = 'Please send me the picture as a \'Photo\', not as a \'File\'.')

def help(bot, update):
	print('help command')
	bot.sendMessage(update.message.chat_id, text = 'Пожалуйста отправьте мне фото, которое необходимо обработать. \nДля отправки фото выберите команду \start или просто отправьте фото в сообщении.')
	

def status(bot, update):
	print('Status command')
	bot.sendMessage(Update.message.chat_id, text = 'Telegram Bot v1.0, \nStatus - OK')

def start(bot, update):
	print('Start command')
	bot.sendMessage(update.message.chat_id, text = 'Добро пожаловать в Object detector bot. \nПожалуйста отправьте мне фото, которое необходимо обработать.')

def run_bot():
	updater = Updater('542863707:AAE7AjGg3iEun00rpgphkyPerJ_cDrawDDU')

	dp = updater.dispatcher
	dp.add_handler(CommandHandler('help', help))
	dp.add_handler(CommandHandler('status', status))
	dp.add_handler(CommandHandler('start', start))
	dp.add_handler(MessageHandler([Filters.photo], detect))
	dp.add_handler(MessageHandler([Filters.document], file_error))
	#dp.add_handler(CommandHandler('convert', convert))

	updater.start_polling()
	updater.idle()

if __name__ == '__main__':
	run_bot()
