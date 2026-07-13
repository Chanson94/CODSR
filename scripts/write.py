import os


directory_path = ['/data2/chenhao/Datasets/LSDIR/data','/data2/chenhao/Datasets/FFHQ']
output_file_path = 'scripts/traindata.txt'

with open(output_file_path, 'w') as output_file:
    for directory in directory_path:
        for root, dirs, files in sorted(os.walk(directory)):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                output_file.write(file_path + '\n')

print(f'All file paths written to {output_file_path}')