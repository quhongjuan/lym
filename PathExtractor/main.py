import json
import os

from datasetprocess.CodeApiExtractor import CodeApiExtractor
from datasetprocess.CodeBagOfWordExtractor import CodeBagOfWordExtractor
from datasetprocess.CodeCfgPathsExtractor import CodeCfgPathsExtractor
from datasetprocess.DocStringExtractor import DocStringExtractor
from datasetprocess.FuncNameExtractor import FuncNameExtractor


def process_cfg(input_file, output_file, need_method_name):
    file_input = open(input_file, 'r', encoding='utf-8')
    file_output = open(output_file, 'w', encoding='utf-8')
    id_num = 0
    for line in file_input.readlines():
        id_num += 1
        print(id_num)
        code = line.strip()
        cfg_path = CodeCfgPathsExtractor.extract(code, need_method_name)

        file_output.write(json.dumps(cfg_path) + '\n')


if __name__ == '__main__':
    code = '''
    public int FindProc(String id){
        int i=0; 
        while(i<procs.size()){
            ProcedureEntry pe=procs.elementAt(i);
            if(pe.name.equals(id)){
                return i1;}
            i=i+1;}
        return i2;}
    '''
    # 1,2,3,4,5,7
    # 1,2,3,4,5
    # 1,2,3,4,5,6
    cfg_path = CodeCfgPathsExtractor.extract(code, True)
    for item in cfg_path:
        print(item)
    # print(json.dumps(cfg_path) + '\n')
    # need_method_name = True # 是否需要方法名
    # # dir = 'D:\\code\\data_RQ1'
    # dir = 'D:\\000准备毕设\\transfomer对应的数据集\\data\\java'
    # # data_set = ['train', 'valid', 'test']
    # data_set = ['dev', 'test']
    # for item in data_set:
    #     # input_dir = os.path.join(dir, item, item+'.token.code')
    #     input_dir = os.path.join(dir, item, 'code.original')
    #     # output_dir = os.path.join(dir, item, item+'.token.path')
    #     output_dir = os.path.join(dir, item, item+'.token.path')
    #     process_cfg(input_dir, output_dir, need_method_name)
