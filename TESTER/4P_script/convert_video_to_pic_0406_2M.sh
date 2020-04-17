#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 29 30
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 31 32
python 0.preview_mp4.py $1 ./$2/135_45/ 2 33 34
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 35 36
python 0.preview_mp4.py $1 ./$2/67.5/ 4 39 40
python 0.preview_mp4.py $1 ./$2/45/ 5 41 42
python 0.preview_mp4.py $1 ./$2/22.5/ 6 44 45

