#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 153 154

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 156 157 

python 0.preview_mp4.py $1 ./$2/135_45/ 2 158 160

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 163 164

python 0.preview_mp4.py $1 ./$2/67.5/ 4 167 169

python 0.preview_mp4.py $1 ./$2/45/ 5 170 171

python 0.preview_mp4.py $1 ./$2/22.5/ 6 173 175

