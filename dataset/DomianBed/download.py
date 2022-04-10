# modified from here
# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py

import os
import gdown
import tarfile
from zipfile import ZipFile
import argparse
import shutil
import uuid

# utils #######################################################################

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

# def download_vlcs(data_dir):
#     # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
#     full_path = stage_path(data_dir, "VLCS")

#     download_and_extract("https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8",
#                          os.path.join(data_dir, "VLCS.tar.gz"))

def download_vlcs(data_dir):
    full_path = stage_path(data_dir, "VLCS")

    tmp_path = os.path.join(full_path, "tmp/")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    with open("./dataset/DomianBed/vlcs_files.txt", "r") as f:
        lines = f.readlines()
        files = [line.strip().split() for line in lines]

    download_and_extract("http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
                         os.path.join(tmp_path, "voc2007_trainval.tar"))

    download_and_extract("https://drive.google.com/uc?id=1I8ydxaAQunz9R_qFFdBFtw6rFTUW9goz",
                         os.path.join(tmp_path, "caltech101.tar.gz"))

    download_and_extract("http://groups.csail.mit.edu/vision/Hcontext/data/sun09_hcontext.tar",
                         os.path.join(tmp_path, "sun09_hcontext.tar"))

    tar = tarfile.open(os.path.join(tmp_path, "sun09.tar"), "r:")
    tar.extractall(tmp_path)
    tar.close()

    for src, dst in files:
        class_folder = os.path.join(data_dir, dst)

        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        dst = os.path.join(class_folder, uuid.uuid4().hex + ".jpg")

        if "labelme" in src:
            # download labelme from the web
            gdown.download(src, dst, quiet=False)
        else:
            src = os.path.join(tmp_path, src)
            shutil.copyfile(src, dst)

    shutil.rmtree(tmp_path)

def download_pacs(data_dir):    
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                        os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
            full_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, required=False, default= "./dataset")
    parser.add_argument('--dataset', type=str, default= "vlcs", help='type of experiment: vlcs, pacs')
    args = parser.parse_args()

    if args.dataset == 'vlcs':
        download_vlcs(args.data_dir)
    if args.dataset == 'pacs':
        download_pacs(args.data_dir)
