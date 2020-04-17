#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 141 142 
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 142 143 
python 0.preview_mp4.py $1 ./$2/135_45/ 2   144 145
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 146 147
python 0.preview_mp4.py $1 ./$2/67.5/ 4 151 152
python 0.preview_mp4.py $1 ./$2/45/ 5 153 154
python 0.preview_mp4.py $1 ./$2/22.5/ 6 154 155

