#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0  126 127 
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 127 128
python 0.preview_mp4.py $1 ./$2/135_45/ 2 129 130 
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 131 132
python 0.preview_mp4.py $1 ./$2/67.5/ 4 134 135
python 0.preview_mp4.py $1 ./$2/45/ 5 137 138
python 0.preview_mp4.py $1 ./$2/22.5/ 6 141 142 

