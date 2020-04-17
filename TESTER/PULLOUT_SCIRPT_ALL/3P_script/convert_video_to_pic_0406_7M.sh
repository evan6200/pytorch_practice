#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 149 150
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 151 152
python 0.preview_mp4.py $1 ./$2/135_45/ 2  154 155 
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3  156 157
python 0.preview_mp4.py $1 ./$2/67.5/ 4 160 161
python 0.preview_mp4.py $1 ./$2/45/ 5 162 163
python 0.preview_mp4.py $1 ./$2/22.5/ 6 164 165

