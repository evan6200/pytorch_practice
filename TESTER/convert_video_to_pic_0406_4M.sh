#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 6 7

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 8 13

python 0.preview_mp4.py $1 ./$2/135_45/ 2 17 24

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 27 31

python 0.preview_mp4.py $1 ./$2/67.5/ 4 37 42

python 0.preview_mp4.py $1 ./$2/45/ 5 47 51

python 0.preview_mp4.py $1 ./$2/22.5/ 6 55 60

