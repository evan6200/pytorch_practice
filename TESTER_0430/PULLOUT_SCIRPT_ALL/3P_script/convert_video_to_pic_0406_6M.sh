#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 123 124
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 125 126
python 0.preview_mp4.py $1 ./$2/135_45/ 2 127 129
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 131 132
python 0.preview_mp4.py $1 ./$2/67.5/ 4 135 136
python 0.preview_mp4.py $1 ./$2/45/ 5 138 139
python 0.preview_mp4.py $1 ./$2/22.5/ 6 143 144

