#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 212 213

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 224 225

python 0.preview_mp4.py $1 ./$2/135_45/ 2 218 219

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 221 222

python 0.preview_mp4.py $1 ./$2/67.5/ 4 227 228

python 0.preview_mp4.py $1 ./$2/45/ 5 230 231

python 0.preview_mp4.py $1 ./$2/22.5/ 6 234 236

