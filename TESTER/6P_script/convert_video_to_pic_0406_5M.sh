#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 82 83
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 83 84 
python 0.preview_mp4.py $1 ./$2/135_45/ 2 85 86
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 88 89
python 0.preview_mp4.py $1 ./$2/67.5/ 4 92 93
python 0.preview_mp4.py $1 ./$2/45/ 5  94 95
python 0.preview_mp4.py $1 ./$2/22.5/ 6 97 98


