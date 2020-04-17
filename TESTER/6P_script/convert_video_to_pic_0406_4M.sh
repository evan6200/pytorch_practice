#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 62 63 
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 64 65
python 0.preview_mp4.py $1 ./$2/135_45/ 2 66 67
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 68 69
python 0.preview_mp4.py $1 ./$2/67.5/ 4 70 71
python 0.preview_mp4.py $1 ./$2/45/ 5 73 75
python 0.preview_mp4.py $1 ./$2/22.5/ 6 76 77

