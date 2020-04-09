#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 65 66

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 67 69

python 0.preview_mp4.py $1 ./$2/135_45/ 2 71 73

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 75 81

python 0.preview_mp4.py $1 ./$2/67.5/ 4 86 90

python 0.preview_mp4.py $1 ./$2/45/ 5 92 98

python 0.preview_mp4.py $1 ./$2/22.5/ 6 102 107

