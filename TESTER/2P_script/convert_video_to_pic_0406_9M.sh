#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 231 232

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 233 234

python 0.preview_mp4.py $1 ./$2/135_45/ 2 235 237

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 241 242

python 0.preview_mp4.py $1 ./$2/67.5/ 4 245 246

python 0.preview_mp4.py $1 ./$2/45/ 5 247 248

python 0.preview_mp4.py $1 ./$2/22.5/ 6 250 252

