#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0  24 25 
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 25 26 
python 0.preview_mp4.py $1 ./$2/135_45/ 2 27 28
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3  29 30
python 0.preview_mp4.py $1 ./$2/67.5/ 4 33 34 
python 0.preview_mp4.py $1 ./$2/45/ 5 35 36
python 0.preview_mp4.py $1 ./$2/22.5/ 6 38 39 

