'''
문3) ~/chap03_Numpy/data 폴더에 포함된 전체 텍스트 파일(*.txt) 전체를 읽어서 리스트에 저장하시오. 

    (text 파일 읽기 형식) 
    file = open(file, mode='r', encoding='utf-8') 
'''


from glob import glob # 파일 검색 패턴 사용

# text file 경로 
path = r"C:\ITWILL\4_Python_ML\workspace\chap03_Numpy" # 파일 기본 경로 

full_text = [] # 텍스트 저장 list 


for file_name in glob(path + '/data/*.txt') :
    file = open(file_name, mode='r', encoding='utf-8')
    # file read & save
    full_text.append(file.read()) 
    file.close()
    
print(full_text)    
    
    
    
    
    
    
    
    


