#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 56 57

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 58 59

python 0.preview_mp4.py $1 ./$2/135_45/ 2 62 63

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 66 68

python 0.preview_mp4.py $1 ./$2/67.5/ 4 72 73

python 0.preview_mp4.py $1 ./$2/45/ 5 74 75

python 0.preview_mp4.py $1 ./$2/22.5/ 6 77 78

