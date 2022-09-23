# cnn_Landsat_time_series_classification_v2-Python

Single pixel s*t Landsat time series classification using 1D CNN

Sep 22, 2022 update (version 2): 
The 1D CNN classification codes are available at https://github.com/hankui/cnn_Landsat_time_series_classification_v2-Python

The NLCD training data is available at 10.5281/zenodo.7106054

The NLCD training data is derived from Landsat 5/7 analysis ready data (ARD) in year 2011 (as x predictor variable) and National Land Cover Database (NLCD) 2011 (as y response variable)

The NLCD training data is distributed across Continental United States (CONUS) with 3,314,439 30m pixel locations

The NLCD training data include (i) NLCD label with 15 classes, i.e., all NLCD classes except ice (https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description)
	(ii) year 2011 growing season Landsat ARD percentiles for Landsat 5/7 bands 2, 3, 4, 5 and 7 and for 8 band ratios derived from the five bands 
	(iii) percentiles include 10th, 20th, 25th, 30th, 35th, 40th, 50th (median), 60th, 65th, 70th, 75th, 80th, 90th so that 
		one can use 5 percentiles (10th, 25th, 50th, 75th, and 90th)
					7 percentiles (10th, 20th, 35th, 50th, 65th, 80th, and 90th)
					9 percentiles (10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, and 90th)
	(iv) the pixel location represented in Landsat ARD tile h and v no. and the pixel i and j locations in the tile
	(v) the no. of the cloud free observations in 2011 growing season derived for the pixel location
  

#*************************************************************************************************************#

A munuscript describing how the data were derived and how the 1D CNN was adapted to the data is in review 


#*************************************************************************************************************#

The codes were written in python (v3.7) and tensorflow (v2.6). 

The parameters are:

(1) learning rate: cnn training initial learning rate 0.01 used in the paper 

(2) epoch: cnn training epochs 70 used in the paper 

(3) method: cnn training optimizer method 1: Adam method 2: dynamic learning rate used in the paper

(4) L2: L2 regularization value; 0.001 used in paper 

(5) layer: no. of CNN layers (can be 4, 5 and 8) and 5 and 8 used in the paper

(6) perc: training data percentages (can be 0.1, 0.5 and 0.9) tested in the paper; the evaluation is used the left 10% 

(7) gpui: which gpu process it will use (only applicable with multi-gpus) 

(8) IMG_HEIGHT: the no. of percentiles (can be 3, 5, 7 and 9) and 5, 7 and 9 used in the paper 

An example would be: 

version=7_4 

layer=5; perc=0.1; gpui=0;IMG_HEIGHT=5

method=0; learning_rate=0.01;   epoch=10; iter=1; L2=0.001; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 

echo "python Pro_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${layer} ${perc} ${gpui} ${IMG_HEIGHT} "

python Pro_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${layer} ${perc} ${gpui} ${IMG_HEIGHT} > layer${layer}.p${perc}.d${IMG_HEIGHT}.rate${learning_rate}.e${epoch}.L${L2}.v${version} & 


#*************************************************************************************************************#

Aug 29, 2021 (version 1): 
Training data: There are 2 input text files (csv) storing the 3,314,439 NLCD and 484,476 CDL land cover training samples:
    NLCD training: ./NLCD/metric.ard.nlcd.Mar01.18.40.txt
    CDL training: ./CDL/metric.ard.nlcd.Mar01.18.40.txt

The codes and their usages are at: 
	https://github.com/hankui/cnn_Landsat_time_series_classification_v1-R
