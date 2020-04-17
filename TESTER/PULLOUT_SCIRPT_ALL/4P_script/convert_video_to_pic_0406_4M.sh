#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 73 74
python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 75 76  
python 0.preview_mp4.py $1 ./$2/135_45/ 2 77 78
python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 78 79
python 0.preview_mp4.py $1 ./$2/67.5/ 4 81 82
python 0.preview_mp4.py $1 ./$2/45/ 5 86 87
python 0.preview_mp4.py $1 ./$2/22.5/ 6 91 92

