#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 67 68
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 69 70
python 0.preview_mp4.py $1 ./$2/135_45/ 2 73 74
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 76 77
python 0.preview_mp4.py $1 ./$2/67.5/ 4 85 88
python 0.preview_mp4.py $1 ./$2/45/ 5 90 93
python 0.preview_mp4.py $1 ./$2/22.5/ 6 95 96

