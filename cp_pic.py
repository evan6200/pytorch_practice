import os
import sys
from shutil import copyfile

print 'usage:'
print '                                            start end'
print 'python cp_pic.py /home/evan/mp4_to_png/9m/90 100 140'

video_image='/home/evan/mp4_to_png/cut_video/'
copy_to=sys.argv[1]+'/'
from_num_pic=sys.argv[2]
to_num_pic=sys.argv[3]

files=[name for name in os.listdir(video_image)]
files.sort()
#os.rename('a.txt', 'b.kml')

for index,filename in enumerate(files):
    dot=filename.find('.')
    pic_name=filename[:dot]
    #print 'from_num_pic',from_num_pic, 'to_num_pic',to_num_pic
    if int(from_num_pic) <= int(pic_name) and int(to_num_pic) >= int(pic_name):
        src=video_image+filename
        dst=copy_to+filename
        print 'src',src,'dst',dst
        copyfile(src, dst)
    #copyfile(src, dst)
    #start_str='%05d'% start
    #replaced_name=filename.replace(filename[:dot],start_str)
    #orig_name=add_image+filename
    #modified_name=add_image+replaced_name
    #os.rename(orig_name, modified_name)
    #start=start+1
