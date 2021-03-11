#!/usr/bin/env python
""" Downloads FaceForensics public data release
Example usage:
    # source-to-target
    python download.py -d \<'compressed' or 'raw'> \<output folder>   
    # self-reenactment 
    python download.py -d \<'selfreenactment_compressed' or 'selfreenactment_raw'> \<output folder>  
    # cropped self-reenactment images
    python download.py -d selfreenactment_images' \<output folder>  
    # only original videos
    python download.py -d original_videos \<output filename>
"""
# -*- coding: utf-8 -*-
import argparse
import os
import urllib
import urllib.request
import tempfile
from os.path import join


SERVER_URL = 'http://kaldir.vc.in.tum.de/FaceForensics/'
TOS_URL = SERVER_URL + 'webpage/FaceForensics_TOS.pdf'
BASE_URL = SERVER_URL + 'v1_cargo/'
ORIGINAL_VIDEOS_URL = BASE_URL + 'original_videos.tar.gz'
RELEASE_DATASET_SIZE = {'raw': '~3.5TB',
                        'compressed': '~130GB',
                        'images': '~135GB'}
DATASET_TYPES = ["raw", "compressed", "selfreenactment_raw",
                 "selfreenactment_compressed", "original_videos",
                 "selfreenactment_images", "source_to_target_images"]
NUM_SAMPLES=5


def get_filelist(filelist_url):
    lines = urllib.request.urlopen(filelist_url)
    video_filenames = []
    for line in lines:
        line = line.decode('utf-8')
        video_filename = line.rstrip('\n')
        video_filenames.append(video_filename)
    return video_filenames


def download_files(filenames, base_url, output_path, sample_only=False):
    os.makedirs(output_path, exist_ok=True)
    num_filenames=len(filenames) if not sample_only else NUM_SAMPLES
    for i, filename in enumerate(filenames):
        if i % 10 == 0:
            print("{}/{}".format(i, num_filenames))
        download_file(base_url + filename, join(output_path, filename))
        if sample_only and i != 0 and i % (NUM_SAMPLES - 1) == 0:
            break
    print("{}/{}".format(num_filenames, num_filenames))


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)


def main():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics public data release.')
    parser.add_argument('output_path', help='directory in which to download')
    parser.add_argument('-d', '--dataset_type', default='compressed',
                        help='Enter which dataset you want to download: '\
                             '"raw", "compressed", "selfreenactment_raw", ' \
                             '"selfreenactment_compressed", ' \
                             '"original_videos" ' \
                             '"source_to_target_images" or ' \
                             '"selfreenactment_images".')
    parser.add_argument('--not_altered', action='store_true',
        help="don't download face2face altered videos")
    parser.add_argument('--not_original', action='store_true',
        help="don't download original videos")
    parser.add_argument('--not_mask', action='store_true',
        help="don't download face2face mask videos")
    parser.add_argument('--not_test', action='store_true',
        help="don't download videos of the test set")
    parser.add_argument('--not_train', action='store_true',
        help="don't download videos of the training set")
    parser.add_argument('--not_val', action='store_true',
        help="don't download videos of the validation set")
    parser.add_argument('--sample_only', action='store_true',
        help='activate this, if you only want to download 5 files per '
             'subfolder')
    args = parser.parse_args()

    # Check for dataset type
    if args.dataset_type not in DATASET_TYPES:
        raise Exception('Wrong dataset type. Please consult "-h" for possible'
                        'options.')

    # TOS
    print('By pressing any key to continue you confirm that you have agreed '\
          'to the FaceForensics terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    key = input('')

    # Check which videos to download
    downloaded_video_types = []
    if not args.not_altered: downloaded_video_types.append('altered')
    if not args.not_original: downloaded_video_types.append('original')
    if not 'images' in args.dataset_type:
        if not args.not_mask: downloaded_video_types.append('mask')

    # Check which folders to download
    downloaded_folders = []
    if not args.not_test: downloaded_folders.append('test')
    if not args.not_train: downloaded_folders.append('train')
    if not args.not_val: downloaded_folders.append('val')

    # Check for dataset type
    if 'selfreenactment' in args.dataset_type:
        dataset = 'selfreenactment'
        dataset_type = args.dataset_type.replace('selfreenactment_', '')
    else:
        dataset = 'source_to_target'
        dataset_type = args.dataset_type.replace('source_to_target_', '')

    # Warning
    if not args.dataset_type == 'original_videos':
        dataset_filesize = RELEASE_DATASET_SIZE[dataset_type]
        print('***')
        if not args.sample_only:
            print('WARNING: You are downloading the FaceForensics dataset {} of'
                  ' size {}'.format(args.dataset_type, dataset_filesize))
        print(
            'Note that existing scan directories will be skipped. Delete ' \
            'partially downloaded directories to re-download.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')

    # Download
    print('\nDownloading dataset: {}'.format(args.dataset_type))
    if args.dataset_type == 'original_videos':
        print('Please be patient, this may take a while (~2gb)')
        download_file(ORIGINAL_VIDEOS_URL,
                      out_file=join(args.output_path,
                                    'faceforensics_original_videos.tar.gz'))
    else:
        for folder in downloaded_folders:
            if 'images' in args.dataset_type:
                filelist_folder = 'images_' + folder
            else:
                filelist_folder = folder
            filelist_url = BASE_URL + '{}/filelists/{}.txt'.format(dataset,
                                                                filelist_folder)
            filenames = get_filelist(filelist_url)
            print('\nDownloading {}'.format(folder))
            for video_type in downloaded_video_types:
                output_path = join(args.output_path,
                                   'FaceForensics_{}'.format(args.dataset_type),
                                   folder, video_type)
                print('{}/{} > {}'.format(folder, video_type, output_path))
                base_url = BASE_URL + '{}/{}/{}/{}/'.format(dataset,
                                                            dataset_type,
                                                            folder,video_type)
                download_files(filenames, base_url, output_path=output_path,
                               sample_only=args.sample_only)


if __name__ == "__main__":
    main()