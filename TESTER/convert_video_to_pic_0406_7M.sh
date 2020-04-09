#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 213 214

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 214 219

python 0.preview_mp4.py $1 ./$2/135_45/ 2 222 225

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 228 232

python 0.preview_mp4.py $1 ./$2/67.5/ 4 238 243

python 0.preview_mp4.py $1 ./$2/45/ 5 248 253

python 0.preview_mp4.py $1 ./$2/22.5/ 6 258 263

