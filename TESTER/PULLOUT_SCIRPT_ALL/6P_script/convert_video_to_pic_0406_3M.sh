#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 43 44
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1  44 45 
python 0.preview_mp4.py $1 ./$2/135_45/ 2 46 47
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 49 50
python 0.preview_mp4.py $1 ./$2/67.5/ 4 54 55
python 0.preview_mp4.py $1 ./$2/45/ 5 55 56 
python 0.preview_mp4.py $1 ./$2/22.5/ 6 57 58

