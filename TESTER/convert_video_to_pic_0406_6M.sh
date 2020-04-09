#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 144 145

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 147 153

python 0.preview_mp4.py $1 ./$2/135_45/ 2 158 162

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 164 171

python 0.preview_mp4.py $1 ./$2/67.5/ 4 176 186

python 0.preview_mp4.py $1 ./$2/45/ 5 192 196

python 0.preview_mp4.py $1 ./$2/22.5/ 6 199 205

