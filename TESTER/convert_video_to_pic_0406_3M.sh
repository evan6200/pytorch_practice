#!/bin/bash
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 117  119

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 120 122

python 0.preview_mp4.py $1 ./$2/135_45/ 2 123 127

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 136 141

python 0.preview_mp4.py $1 ./$2/67.5/ 4 148 153

python 0.preview_mp4.py $1 ./$2/45/ 5 157 163

python 0.preview_mp4.py $1 ./$2/22.5/ 6 168 174

