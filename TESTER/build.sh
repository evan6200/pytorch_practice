#!/bin/bash
echo $#
if [ $# -le 0 ] ; then
   echo 'sh build.sh [inject_FIEL_NAME]'
   echo 'example sh build .sh 1P_1M_' 
   exit 0
fi
mkdir $1'image'
sh collect_all.sh $1'image/'
sh mark_distance.sh $1 $1'image/'
