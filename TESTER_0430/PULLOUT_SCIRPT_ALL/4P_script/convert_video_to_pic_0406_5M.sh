#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 98 99
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 100 101 
python 0.preview_mp4.py $1 ./$2/135_45/ 2 102 103
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 104 105
python 0.preview_mp4.py $1 ./$2/67.5/ 4 108 109
python 0.preview_mp4.py $1 ./$2/45/ 5 110 111
python 0.preview_mp4.py $1 ./$2/22.5/ 6 112 113

