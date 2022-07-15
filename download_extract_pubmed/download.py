import os
files_name = [f'pubmed22n{str(i).zfill(4)}.xml.gz' for i in range(1,1115)]

for file_name in files_name:
  os.system(f'wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{file_name}')
  os.system(f'gzip -d {file_name}')