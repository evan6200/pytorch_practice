#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0  180 181
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 182 183
python 0.preview_mp4.py $1 ./$2/135_45/ 2 184 185
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 187 188
python 0.preview_mp4.py $1 ./$2/67.5/ 4 191 192
python 0.preview_mp4.py $1 ./$2/45/ 5 193 194
python 0.preview_mp4.py $1 ./$2/22.5/ 6 196 197

