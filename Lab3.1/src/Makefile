# NOTE: The variables PATH_INCLUDES & PATH_LIB must match the location of your OpenCV installation  
#
# To set PATH_INCLUDE: 
#	To find the 'include' directory, type in a terminal:
#	$ find / -name opencv.hpp 2>&1 | grep -v "Permission denied"
#	$ /usr/local/include/opencv2/opencv.hpp
#	so your 'include' directory is located in '/usr/local/include/'
#
# To set PATH_LIB: 
#	To find the 'lib' directory, type in a terminal:
#	$ find / -name libopencv* 2>&1 | grep -v "Permission denied"
#	$ ...
#	$ /usr/local/lib/libopencv_videostab.so.3.0
#	$ ...
#	$ /usr/local/lib/libopencv_core.so.3.0.0
#	$ /usr/local/lib/libopencv_core.so.3.0
#	$ /usr/local/lib/libopencv_core.so
#	$ ...
#	so your 'lib' directory is located in '/usr/local/lib/'
#	and your OpenCV version is 3.0.0
#
#   In this sample code, OpenCV is installed in the path '/opt/installation'
#	with the following details:
#		Version: 3.4.4
#		Include Path: /opt/installation/OpenCV-3.4.4/include
#		Library Path: /opt/instllation/OpenCV-3.4.4/lib
#
#	Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)

LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_videoio -lopencv_imgcodecs -lopencv_flann
PATH_INCLUDES = /opt/installation/OpenCV-3.4.4/include
PATH_LIB = /opt/installation/OpenCV-3.4.4/lib

all: Lab3.1

Lab3.1: Lab3.1.o blobs.o ShowManyImages.o
	g++ Lab3.1.o blobs.o ShowManyImages.o -o Lab3.1 -L$(PATH_LIB) $(LIBS) -lm

Lab3.1.o: Lab3.1.cpp
	g++ -c Lab3.1.cpp -I$(PATH_INCLUDES) -O

blobs.o: blobs.cpp
	g++ -c blobs.cpp -I$(PATH_INCLUDES) -O

ShowManyImages.o: ShowManyImages.cpp
	g++ -c ShowManyImages.cpp -I$(PATH_INCLUDES) -O



