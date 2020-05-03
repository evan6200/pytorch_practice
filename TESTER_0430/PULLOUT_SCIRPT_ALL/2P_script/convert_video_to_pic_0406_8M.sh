#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 210 211

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 212 213

python 0.preview_mp4.py $1 ./$2/135_45/ 2 214 215

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 217 218

python 0.preview_mp4.py $1 ./$2/67.5/ 4 222 223

python 0.preview_mp4.py $1 ./$2/45/ 5 225 226

python 0.preview_mp4.py $1 ./$2/22.5/ 6 227 228

