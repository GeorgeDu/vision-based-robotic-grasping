import os
from tqdm import tqdm

def read_MD_pdf(old_filename, new_filename):
    """
    NOTE: this function is for changing the line with 'arxiv.org/abs' into that with '.pdf'.
    :param old_filename: README.md
    :param new_filename: README_pdf.md
    :return:
    """
    f = open(old_filename, 'r', encoding='utf-8')
    f_new = open(new_filename, 'w+', encoding='utf-8')
    for line in tqdm(f.readlines()):
        if line.find('arxiv.org/abs') != -1:
            line = line.replace('arxiv.org/abs', 'arxiv.org/pdf')
            line = line.replace(')]', '.pdf)]', 1)
        f_new.write(line)
    f.close()
    f_new.close()


if __name__ == '__main__':
    read_MD_pdf('README.md', 'README_pdf.md')
