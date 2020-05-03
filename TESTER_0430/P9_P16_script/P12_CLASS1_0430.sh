#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "$angle_folder /home/evan/gopro_pic/2020.04.29/GOPR0273.MP4"

#Person Number = 9

CLASS0="/home/evan/gopro_pic/2020.04.29/GOPR0273.MP4"
CLASS=$CLASS0
label_class='1'
angle_folder='112_22.5'
person='12'
#9M
python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 1
python rename_key_point.py  ./$angle_folder/ ${person}P_9M_
mkdir -p ${person}P_tester/${person}P_9M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_9M_image
mv ./$angle_folder/* ALL_TEST_0430/


python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 8
python rename_key_point.py  ./$angle_folder/ ${person}P_8M_
mkdir -p ${person}P_tester/${person}P_8M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_8M_image
mv ./$angle_folder/* ALL_TEST_0430/

python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 15
python rename_key_point.py  ./$angle_folder/ ${person}P_7M_
mkdir -p ${person}P_tester/${person}P_7M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_7M_image
mv ./$angle_folder/* ALL_TEST_0430/

python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 21
python rename_key_point.py  ./$angle_folder/ ${person}P_6M_
mkdir -p ${person}P_tester/${person}P_6M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_6M_image
mv ./$angle_folder/* ALL_TEST_0430/

python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 29
python rename_key_point.py  ./$angle_folder/ ${person}P_5M_
mkdir -p ${person}P_tester/${person}P_5M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_5M_image
mv ./$angle_folder/* ALL_TEST_0430/

python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 38
python rename_key_point.py  ./$angle_folder/ ${person}P_4M_
mkdir -p ${person}P_tester/${person}P_4M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_4M_image
mv ./$angle_folder/* ALL_TEST_0430/

python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 45
python rename_key_point.py  ./$angle_folder/ ${person}P_3M_
mkdir -p ${person}P_tester/${person}P_3M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_3M_image
mv ./$angle_folder/* ALL_TEST_0430/

python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 53
python rename_key_point.py  ./$angle_folder/ ${person}P_2M_
mkdir -p ${person}P_tester/${person}P_2M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_2M_image
mv ./$angle_folder/* ALL_TEST_0430/

python 0.preview_mp4.py $CLASS ./$angle_folder/ $label_class 61
python rename_key_point.py  ./$angle_folder/ ${person}P_1M_
mkdir -p ${person}P_tester/${person}P_1M_image
cp ./$angle_folder/* ${person}P_tester/${person}P_1M_image
mv ./$angle_folder/* ALL_TEST_0430/


#python 0.preview_mp4.py $1 ./$2/112_22.5/ 1  169 170
#python 0.preview_mp4.py $1 ./$2/135_45/ 2 171 174
#python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 176 177
#python 0.preview_mp4.py $1 ./$2/67.5/ 4 180 181
#python 0.preview_mp4.py $1 ./$2/45/ 5 182 183
#python 0.preview_mp4.py $1 ./$2/22.5/ 6 184 185

