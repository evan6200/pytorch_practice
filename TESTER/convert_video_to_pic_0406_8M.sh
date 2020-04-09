#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 269 274

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 288 293

python 0.preview_mp4.py $1 ./$2/135_45/ 2 277 283

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 301 306

python 0.preview_mp4.py $1 ./$2/67.5/ 4 311 316

python 0.preview_mp4.py $1 ./$2/45/ 5 317 324

python 0.preview_mp4.py $1 ./$2/22.5/ 6 328 333

