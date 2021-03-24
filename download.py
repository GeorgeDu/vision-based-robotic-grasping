# -*- coding: utf-8 -*-
# @Time : 2021/3/24
# @Author : Wen Hongtao
# @Email : hatimwen@163.com
# @File : download.py
# @Software: PyCharm


import re
import os
import wget
from tqdm import tqdm
from pathlib import Path

# Here to set the Md file you use for download papers.
# And you can split the original file into multiple parts,
# so you can pay more attention to the papers you need most.
name = '6DoF Grasp.md'


def getDownloadAddress(_str):
    add = re.findall(r'htt.*pdf', _str)[0]
    return add


def getPaperName(_str):
    papername = re.findall(r']\*\*(.*)\[\[paper]', _str)[0] \
        .strip().strip(',')
    return papername


def downloadFile(filename, add, success, fail_list):
    """
    :param filename: the filename including its filepath.
    :param add: the file's download address.
    :param success: the last success value.
    :param fail_list: the list of failed ones.
    :return: success, fail_list
    """

    try:
        file_name = wget.download(add, out=filename)
    except:
        print('\n****************************Net error'
              '****************************')
        print('Can\'t download {0} from: {1}'.format(filename, add))
        fail_list.append(filename)
        fail_list.append(add)
        return success, fail_list
    else:
        success += 1
        print('\nDownloaded in:' + file_name)
    return success, fail_list


# get MD file and download file.pdf
def read_down_MD(name):
    """

    :param name: the name of MD file.
    :return: None value
    """
    f = open(name, 'r')
    success = 0
    count = 0
    year = '0'
    # like: 2021
    source_name = 'NotKnown'
    # like: arXiv
    folderpath_2 = ''
    folderpath_3 = ''
    folderpath_4 = ''
    folderpath_5 = ''
    fail_list = []
    for line in tqdm(f.readlines()):
        year_idx = line.find('***20')
        if year_idx != -1:
            year = line[3:7]
        if line.find('## ') == 0:
            folderpath_2 = line[3:].replace('.', '_') \
                .replace('\n', '').replace(' ', '')
            folderpath_3 = ''
            folderpath_4 = ''
            folderpath_5 = ''
        if line.find('### ') == 0:
            folderpath_3 = line[4:].replace('.', '_') \
                .replace('\n', '').replace(' ', '')
            folderpath_4 = ''
            folderpath_5 = ''
        if line.find('#### ') == 0:
            folderpath_4 = line[5:].replace('.', '_') \
                .replace('\n', '').replace(' ', '')
            folderpath_5 = ''
        if line.find('##### ') == 0:
            folderpath_5 = line[6:].replace('.', '_') \
                .replace('\n', '').replace(' ', '')
        folderpath = os.path.join(folderpath_2, folderpath_3,
                                  folderpath_4, folderpath_5)
        if '[[paper](' in line and '**[' in line:
            source_name = re.findall(r'\*\*\[(.*)]\*\*', line)[0]
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            filename = getPaperName(line)
            filename = filename.replace(':', '_').replace('?', '_') \
                .replace('\\', '_').replace('/', '_') \
                .replace('*', '_').replace('"', '_')
            full_name = source_name + year + '_' + filename
            filename = os.path.join(folderpath, full_name + '.pdf')
            if not Path(filename).exists():
                try:
                    add = getDownloadAddress(line)
                except:
                    print('\n**********************Address error'
                          '**********************')
                    print('Can\'t download {0} in line: {1}'
                          .format(filename, line))
                    fail_list.append(filename)
                    fail_list.append(line)
                    count = count + 1
                    continue
                else:
                    success, fail_list = downloadFile(filename, add,
                                                      success, fail_list)
                    count = count + 1
                    continue
            else:
                success += 1
                print('\nAlready downloaded:' + filename)
                count = count + 1
                continue
    f.close()
    print('*************************************************************')
    print("********All {0}/{1} pdfs' download have finished.************"
          .format(success, count))
    if fail_list:
        print("**********************Fail_list: ****************************")
        for fail in fail_list:
            print(fail)
    print('*************************************************************')


if __name__ == '__main__':
    read_down_MD(name)
