#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 80 81 
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 81 82
python 0.preview_mp4.py $1 ./$2/135_45/ 2 83 84
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 86 87
python 0.preview_mp4.py $1 ./$2/67.5/ 4 90 91
python 0.preview_mp4.py $1 ./$2/45/ 5 92 93
python 0.preview_mp4.py $1 ./$2/22.5/ 6 95 96


