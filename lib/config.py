import socket
if 'ccvl' in socket.gethostname():
    data_dir = '/mnt/disk1/vittal/visual_concepts/'
    caffe_dir = '/srv/share/code/caffe/'
else:
    data_dir = '/home-4/vpremac1@jhu.edu/scratch/'
