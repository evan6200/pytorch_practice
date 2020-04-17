#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 32 33

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 34 35

python 0.preview_mp4.py $1 ./$2/135_45/ 2 36 37

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 39 40

python 0.preview_mp4.py $1 ./$2/67.5/ 4 46 47 

python 0.preview_mp4.py $1 ./$2/45/ 5 48 50

python 0.preview_mp4.py $1 ./$2/22.5/ 6 52 53

