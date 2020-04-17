#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 168 169
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1  169 170
python 0.preview_mp4.py $1 ./$2/135_45/ 2 171 174
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 176 177
python 0.preview_mp4.py $1 ./$2/67.5/ 4 180 181
python 0.preview_mp4.py $1 ./$2/45/ 5 182 183
python 0.preview_mp4.py $1 ./$2/22.5/ 6 184 185

