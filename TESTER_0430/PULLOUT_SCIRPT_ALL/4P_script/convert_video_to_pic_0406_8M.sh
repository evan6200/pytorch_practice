#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 169 170 
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 170 171
python 0.preview_mp4.py $1 ./$2/135_45/ 2 171 172 
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 177 178
python 0.preview_mp4.py $1 ./$2/67.5/ 4  183 184
python 0.preview_mp4.py $1 ./$2/45/ 5 186 187
python 0.preview_mp4.py $1 ./$2/22.5/ 6 190 191

