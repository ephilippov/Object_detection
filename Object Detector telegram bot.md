**Object Detector telegram bot.**

Реализация telegram бота для распознавания объектов на фото – [@ObjectDetector\_bot](https://t.me/ObjectDetector_bot).

**!NB** Telegrambot может быть не доступен, тк на текущий момент он развернут на localhost.

Все исходные коды (бот + Caffemodel) доступны на [GitHub](https://github.com/ephilippov/Object_detection).

Для работы бота ему нужно прислать фото (картинку) на которой необходимо распознать объекты. Фото можно отправить в сообщении или же переслать полученное в другом чате.

Пример работы с ботом:

На вход боту отправляется фото

 ![Car img](/images.fld/image001.jpg)

### После обработки полученного фото телеграмм ботом, данное изображение передается на вход нейронной сети, обрабатывается и выдается результат с размеченными объектами.

###

![Detected car](/images.fld/image003.jpg)

### В качестве сети была выбрана [MobileNet](https://arxiv.org/abs/1704.04861) Single Shot Detector из модуля OpenCV. Данная архитектура выбрана тк она отвечает всем основным требования. А именно, оптимально подходит для использования на мобильных платформах и легковесных приложениях.

### Архитектура сети MobileNet:

###

![MobileNet SSD](/images.fld/image005.png)

Первоначально MobileNet SSD обучалась на [COCO (Common Objects in Context)](http://cocodataset.org/) датасете , затем производилась донастройка на датасете [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) что позволило добиться точности 72.7% в метрике качества ранжирования mAP (mean average precision).

MobileNetSSD позволяет обнаруживать 20 классов на входном изображении, включая людей, кошек, собак, птиц, коров, лошадей, овец, автомобили, автобусы, самолеты, лодки, велосипеды, мотоциклы, столы, стулья, диваны, телевизоры и растения.

### Библиотека (OpenCV) выбрана тк она является наиболее известной библиотекой компьютерного зрения, имеет большое комьюнити разработчиков и стала «стандартом» в области компьютерного зрения. Возможность использовать библиотеку на языке Python добавляет большие возможности в области Deep Learning.

В качестве фреймворка используется Caffe. Открытый код и высокая скорость обучения играют большую роль в выборе данного фреймворка из ряда других основных.

### Еще несколько примеров работы сети и телеграмм бота:

###

![](/images.fld/image007.jpg)

###

![](/images.fld/image009.png)

###

![](/images.fld/image011.jpg)