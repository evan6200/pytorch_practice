#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 41 42
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 42 43 
python 0.preview_mp4.py $1 ./$2/135_45/ 2 44 45
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 46 47
python 0.preview_mp4.py $1 ./$2/67.5/ 4 51 52
python 0.preview_mp4.py $1 ./$2/45/ 5 53 54
python 0.preview_mp4.py $1 ./$2/22.5/ 6 55 56

