#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 86 87

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 88 89

python 0.preview_mp4.py $1 ./$2/135_45/ 2 92 95

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 97 99

python 0.preview_mp4.py $1 ./$2/67.5/ 4 104 105

python 0.preview_mp4.py $1 ./$2/45/ 5 106 108

python 0.preview_mp4.py $1 ./$2/22.5/ 6 112 113

