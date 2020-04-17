#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 83 84

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 85 56

python 0.preview_mp4.py $1 ./$2/135_45/ 2 88 86

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 91 92

python 0.preview_mp4.py $1 ./$2/67.5/ 4 94 95

python 0.preview_mp4.py $1 ./$2/45/ 5 96 97

python 0.preview_mp4.py $1 ./$2/22.5/ 6 100 101

