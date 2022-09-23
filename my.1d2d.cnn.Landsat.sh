#!/bin/bash
# nvidia-smi

# bash my.1d2d.cnn.Landsat.sh
# module purge
## if there is gpu
# module load cuda
# module load cudnn
module load python/3.7
module load rasterio
# module load libtiff 
# module load libgeotiff

# module load cuda/11.2.1


# conda init bash 
date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
which python


version=7_1 # final 
version=7_2 # final + test randomness
version=7_4 # final + test randomness

date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
which python
SLEEP=100
# SLEEP=1

layer=5; perc=0.1; gpui=0;IMG_HEIGHT=5
method=0; learning_rate=0.01;   epoch=10; iter=1; L2=0.001; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
echo "python Pro_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${layer} ${perc} ${gpui} ${IMG_HEIGHT} "
python Pro_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${layer} ${perc} ${gpui} ${IMG_HEIGHT} > layer${layer}.p${perc}.d${IMG_HEIGHT}.rate${learning_rate}.e${epoch}.L${L2}.v${version} & 

# gpui=0
# for perc in 0.1 0.5 0.9;
# for perc in 0.9;
# do
    # echo ${perc}
    # for layer in 5 8;
    # for layer in 8;
    # do
		# for IMG_HEIGHT in 3 5 7 9
		# do
        # echo ${perc}
        # echo ${layer}
        # echo ${IMG_HEIGHT}
        # method=0; learning_rate=0.01;   epoch=70; iter=1; L2=0.001; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
		# echo "python Pro_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${layer} ${perc} ${gpui} ${IMG_HEIGHT} "
        # python Pro_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${layer} ${perc} ${gpui} ${IMG_HEIGHT} > layer${layer}.p${perc}.d${IMG_HEIGHT}.rate${learning_rate}.e${epoch}.L${L2}.v${version} & 
		# gpui=$((${gpui}+1))
		# if [ ${gpui} -ge 4 ]; then 
			# wait 
			# gpui=0
			# echo ""
			# echo ""
		# fi
		# done
	# done
	# wait
# done

# wait 
# date_end=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
# time_diff=`echo "scale=2;($date_end-$date_start+0.01)*1.0/3600.0"|bc`;
# date
# echo "$time_diff hours used";
