import os
import sys

print ('usage: replace file name')
print ('python rename_key_point.py [folder_name] [inject str]')

print('sys.argv',len(sys.argv))

if (len(sys.argv)<3):
    print('wrong input, leaving')
    sys.exit()

print('processing...')

path=sys.argv[1]
str_replace=sys.argv[2]
files=[name for name in os.listdir(path)]
files.sort()
for index,filename in enumerate(files):
    dot=filename.find('.')
    new_str=str_replace+filename[:dot]+filename[dot:]
    replaced_name=filename.replace(filename,new_str)     
    orig_name_path=path+filename
    modified_name_path=path+replaced_name
    print (orig_name_path)
    print (modified_name_path)
    os.rename(orig_name_path, modified_name_path)

print('done')
