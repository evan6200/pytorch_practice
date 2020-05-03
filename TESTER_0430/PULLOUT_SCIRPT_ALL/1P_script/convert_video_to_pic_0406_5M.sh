#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 108 109

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 110 111

python 0.preview_mp4.py $1 ./$2/135_45/ 2 112 113

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 116 117

python 0.preview_mp4.py $1 ./$2/67.5/ 4 121 122

python 0.preview_mp4.py $1 ./$2/45/ 5 126 127

python 0.preview_mp4.py $1 ./$2/22.5/ 6 128 130

