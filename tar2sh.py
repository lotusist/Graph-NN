# shell command: time python tar_csv_export.py
import os

#  targets are 3.1 versions, and its range is 2017.10.12 ~ 2019.10.31 . 

PATH = './'
file_target = '_v3.1.tar.gz'
directory_target = './my_code/'


filename_list = []
for (path, dir, files) in os.walk(f'{PATH}'):
    for filename in files:
        if file_target in filename:
            filename_list.append(f'{path}{filename}')
filename_list = sorted(filename_list)

print('the number of target files are : ', len(filename_list))

with open(f'{directory_target}tar_csv_export.sh', 'w') as fp:
    for filename in filename_list:
        folder_name = filename[42:-7]
        fp.write(f'mkdir {PATH}{folder_name}\n')
        fp.write(f'tar -zxvf {filename} -C {PATH}{folder_name}\n')

# after the running the .py file, run the linux code below in the terminal.
# sh ./my_code/tar_csv_export.sh
