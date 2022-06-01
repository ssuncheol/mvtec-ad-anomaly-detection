import os
import shutil

## for using ImageFolder, make sure that each data is inside in its class folder.
path = './mvtec_datafolder/'
class_name = os.listdir(path)

label_list = []
for i in class_name:
    for j in os.listdir(path+i+'/train'):
        label_list.append(i+'_'+j)

train_folder = './mvtec_ad_train/'

for i in label_list:
    if not os.path.exists(train_folder+i):
        os.makedirs(train_folder+i)

for i in class_name:
    for j in os.listdir(path+i+'/train'):
        for k in os.listdir(path+i+'/train/'+j):
            os.system("sudo mv %s %s" % (path+i+'/train/'+j+'/'+k, train_folder+i+'_'+j))
