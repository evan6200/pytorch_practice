#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 10 11

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 12 13

python 0.preview_mp4.py $1 ./$2/135_45/ 2 14 17

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 18 19

python 0.preview_mp4.py $1 ./$2/67.5/ 4 22 23

python 0.preview_mp4.py $1 ./$2/45/ 5 24 25

python 0.preview_mp4.py $1 ./$2/22.5/ 6 26 27

