#!/bin/bash
echo "VIDEO2"
echo "$1=MP4 $2=located_folder"
echo "avi location -> /home/evan/mp4_to_png/0314/avi"
python 0.preview_mp4.py $1 ./$2/90/ 0 339 341

python 0.preview_mp4.py $1 ./$2/112_22.5/ 1 342 347

python 0.preview_mp4.py $1 ./$2/135_45/ 2 351 357

python 0.preview_mp4.py $1 ./$2/157_67.5/ 3 360 366

python 0.preview_mp4.py $1 ./$2/67.5/ 4 371 376

python 0.preview_mp4.py $1 ./$2/45/ 5 382 386

python 0.preview_mp4.py $1 ./$2/22.5/ 6 388 393

