#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 4 5
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 4 5
python 0.preview_mp4.py $1 ./$2/135_45/ 2 6 7
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 8 9
python 0.preview_mp4.py $1 ./$2/67.5/ 4 12 13
python 0.preview_mp4.py $1 ./$2/45/ 5 14 15
python 0.preview_mp4.py $1 ./$2/22.5/ 6 20 21

