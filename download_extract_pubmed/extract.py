import pubmed_parser as pp
import os 
import re

files_name = os.listdir('pubmed')
logs = open('logs.txt', 'w')

 
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
files_name.sort(key=natural_keys)

for file_name in files_name[::-1]:
  print('starting file', file_name)
  out_file_name = file_name.replace('.xml', '.txt')
  if os.path.exists(f'out/{out_file_name}'):
      print('skip', out_file_name)
      continue
  path = f'pubmed/{file_name}'
  dict_out = pp.parse_medline_xml(path)
  print(f'found {len(dict_out)} in {file_name}')
  cnt = 0
  valid = 0
  out_file = f'out/{out_file_name}'
  if os.path.exists(f'out/{out_file_name}'):
      continue
  with open(f'out/{out_file_name}', 'w', encoding='utf-8') as out_file:
    for item in dict_out:
      abstract = item['abstract'].strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
      
      if abstract:
        out_file.write(f'en: {abstract}\n')
        valid += 1
      else:
        cnt += 1
  print(f'invalid abstract: {cnt}')
  print(f'valid abstract: {valid}')  
  logs.write(f'{out_file_name}\n')