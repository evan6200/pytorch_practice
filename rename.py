import os

add_image="/home/evan/add_data/orig/"
files=[name for name in os.listdir(add_image)]
files.sort()
#os.rename('a.txt', 'b.kml')
start=4733

for index,filename in enumerate(files):
    dot=filename.find('.')
    start_str='%05d'% start
    replaced_name=filename.replace(filename[:dot],start_str)     
    orig_name=add_image+filename
    modified_name=add_image+replaced_name
    os.rename(orig_name, modified_name)
    start=start+1
