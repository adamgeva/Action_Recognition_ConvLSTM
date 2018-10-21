

import os
os.system("ls -l")

vid_dir = '/hdd/IR-new/Orig'
vid_dir_stab = '/hdd/IR-new/Stabilized/'


#subdirs = [x[0] for x in os.walk(vid_dir)]
#print(subdirs)


for root, dirs, files in os.walk(vid_dir):
    #folder = root.split('/')[-1]
    #path = vid_dir_stab + folder +'/'
    #os.mkdir(path)
    for name in files:
        vid_in = root + '/' + name
        vid_out = vid_dir_stab + name
        print (vid_in)
        print(vid_out)
        command = 'python3 -m vidstab --input ' + vid_in + ' --output ' + vid_out
        os.system(command)
