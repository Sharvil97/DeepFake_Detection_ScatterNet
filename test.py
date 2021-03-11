'''
author: Christopher Otto
Reference: https://github.com/CatoGit/Comparing-the-Performance-of-Deepfake-Detection-Methods-on-Benchmark-Datasets/blob/master/deepfake_detector/test.py 

Modified By: Sharvil Mainkar
'''
import os
import cv2
import numpy as np
import pandas as pd
from metrics import metrics
import torch
import time
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import glob
import json
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN

from albumentations import Resize
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from facedetector.retinaface import df_retinaface

def vid_inference(model, video_frames, label, , normalization):
    # model evaluation mode
    model.cuda()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_func = nn.BCEWithLogitsLoss()
    #label = torch.from_numpy(label).to(device)
    # get prediction for each frame from vid
    avg_vid_loss = []
    avg_preds = []
    avg_loss = []
    frame_level_preds = []
    for frame in video_frames:
        # turn image to rgb color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize to DNN input size
        resize = Resize(width=256, height=256)
        frame = resize(image=frame)['image']
        frame = torch.tensor(frame).to(device)
        # forward pass of inputs and turn on gradient computation during train
        with torch.no_grad():
            # predict for frame
            # channels first
            frame = frame.permute(2, 0, 1)
            # turn dtype from uint8 to float and normalize to [0,1] range
            frame = frame.float() / 255.0
            # normalize by imagenet stats
            if normalization == 'xception':
                transform = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            elif normalization == "imagenet":
                transform = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            frame = transform(frame)
            # add batch dimension and input into model to get logits
            predictions = model(frame.unsqueeze(0))
            # get probabilitiy for frame from logits
            preds = torch.sigmoid(predictions)
            avg_preds.append(preds.cpu().numpy())
            frame_level_preds.extend(preds.cpu().numpy()[-1])
            # calculate loss from logits
            loss = loss_func(predictions.squeeze(1), torch.tensor(
                label).unsqueeze(0).type_as(predictions))
            avg_loss.append(loss.cpu().numpy())
    # return the prediction for the video as average of the predictions over all frames
    return np.mean(avg_preds), np.mean(avg_loss), frame_level_preds


def inference(model, normalization, dataset, face_margin, num_frames=None):
    running_loss = 0.0
    running_corrects = 0.0
    running_false = 0.0
    running_auc = []
    running_ap = []
    labs = []
    prds = []
    ids = []
    frame_level_prds = []
    frame_level_labs = []
    running_corrects_frame_level = 0.0
    running_false_frame_level = 0.0
    SCALE = None
    # load retinaface face detector
    # net, cfg = df_retinaface.load_face_detector()
    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, thresholds=[0.6, 0.7, 0.99], post_process=False, device=device).eval()
    # Define face extractor
    face_extractor = FaceExtractor(detector=face_detector, n_frames=N_FRAMES,  resize=SCALE)
    test_df = preprocess(data)
    inference_time = time.time()
    print(f"Inference using {num_frames} frames per video.")
    # print(f"Use face margin of {face_margin * 100} %") 
    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        video = row.loc['video']
        label = row.loc['label']
        vid = os.path.join(video)
        # inference (no saving of images inbetween to make it faster)
        # detect faces, add margin, crop, upsample to same size, save to images
        # faces = df_retinaface.detect_faces(net, vid, cfg, num_frames=num_frames)
        # save frames to images
        # try:
        # vid_frames = df_retinaface.extract_frames(
        #     faces, video, save_to=None, face_margin=face_margin, num_frames=num_frames, test=True)

        vid_frames = FaceExtractor(video_path, save_img=False, n_frames=num_frames)
        # if no face detected continue to next video
        if not vid_frames:
            print("No face detected.")
            continue
        # inference for each frame
        # frame level auc can be measured
        vid_pred, vid_loss, frame_level_preds = vid_inference(
            model, vid_frames, label, normalization)
        frame_level_prds.extend(frame_level_preds)
        frame_level_labs.extend([label]*len(frame_level_preds))
        running_corrects_frame_level += np.sum(
            np.round(frame_level_preds) == np.array([label]*len(frame_level_preds)))
        running_false_frame_level += np.sum(
            np.round(frame_level_preds) != np.array([label]*len(frame_level_preds)))
        ids.append(video)
        labs.append(label)
        prds.append(vid_pred)
        running_loss += vid_loss
        # calc accuracy; thresh 0.5
        running_corrects += np.sum(np.round(vid_pred) == label)
        running_false += np.sum(np.round(vid_pred) != label)

    # save predictions to csv for ensembling
    df = pd.DataFrame(list(zip(ids, labs, prds)), columns=[
                      'Video', 'Label', 'Prediction'])

    if dataset is not None:
        df.to_csv(f'{model}_predictions_on_{dataset}.csv', index=False)
    # get metrics
    one_rec, five_rec, nine_rec = metrics.prec_rec(
        labs, prds, model, alpha=100, plot=False)
    auc = round(roc_auc_score(labs, prds), 5)
    frame_level_auc = round(roc_auc_score(
        frame_level_labs, frame_level_prds), 5)
    frame_level_acc = round(running_corrects_frame_level /
                            (running_corrects_frame_level + running_false_frame_level), 5)
    ap = round(average_precision_score(labs, prds), 5)
    loss = round(running_loss / len(prds), 5)
    acc = round(running_corrects / len(prds), 5)
    #save results
    # result = 
    print("Benchmark results:")
    print("Confusion matrix (video-level):")
    # get confusion matrix in correct order
    print(confusion_matrix(np.round(prds), labs, labels=[1, 0]))
    tn, fp, fn, tp = confusion_matrix(labs, np.round(prds)).ravel()
    print(f"Loss: {loss}")
    print(f"Acc: {acc}")
    print(f"AUC: {auc}")
    print(f"AP: {auc}")
    print("Confusion matrix (frame-level):")
    print(confusion_matrix(np.round(frame_level_prds),
                            frame_level_labs, labels=[1, 0]))
    print(f"Frame-level AUC: {frame_level_auc}")
    print(f"Frame-level ACC: {frame_level_acc}")
    print()
    print("Cost (best possible cost is 0.0):")
    print(f"{one_rec} cost for 0.1 recall.")
    print(f"{five_rec} cost for 0.5 recall.")
    print(f"{nine_rec} cost for 0.9 recall.")
    print(
        f"Duration: {(time.time() - inference_time) // 60} min and {(time.time() - inference_time) % 60} sec.")
    print()
    print(
        f"Detected \033[1m {tp}\033[0m true deepfake videos and correctly classified \033[1m {tn}\033[0m real videos.")
    print(
        f"Mistook \033[1m {fp}\033[0m real videos for deepfakes and \033[1m {fn}\033[0m deepfakes went by undetected by the model.")
    if fn == 0 and fp == 0:
        print("Wow! A perfect classifier!")
    return auc, ap, loss, acc

class FaceExtractor:
    def __init__(self, detector, save_img=False, n_frames=None, resize=None):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
#         self.count = count
    
    def __call__(self, filename, save_dir, count):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            save_dir {str} -- The directory where results are saved.
        """

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)
        
        video_frames = []
        # Loop through frames
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                
                if save_img:
                    save_path = os.path.join(save_dir, f'{count}' +"_"+ f'{j}.png')
                
                video_frames.append(fnames)

                self.detector([frame], save_path=save_path)
        v_cap.release()
        return video_frames


def preprocess(data):
        if data == "FaceForensics":
            root = r"data/FaceForensics/test"
        if data == "FaceForensics++":
            root = r"/data/FaceForensics++/test"
        if data == "CelebDF":
            root = r"data/CelebDF/test"
        if data == "GoogleDFD":
            root = r"data/DeepFakeDetection/test"
        if data == "FaceHQ":
            root = r"data/FaceHQ/test"  
        if data == "DFDC":
            root =   r"data/DFDC/test"   
        if data == "DeeperForensics":
            root =   r"data/DeeperForensics/test"   
        if data == "UADFV":
            root =   r"data/UADFV/test"     
        if data == "NeuralTexture":
            root = r"data/NeuralTexture/test"
        if data == "Deepfakes":
            root = r"data/Deepfakes/test"
        if data == "FaceSwap":
            root = r"data/FaceSwap/test"
        if data == "FaceShifter":
            root = r"data/FaceShifter/test"
        if data == "Face2Face":
            root = r"data/Face2Face/test"
        
        real_vids = []
        fake_vids = []
        labels_real = []
        labels_fake = []
        for path in glob.glob(root+"/**/*.mp4", recursive=True):
            if "real" in path:
                real_vids.append(path)
                labels_real.append(1)
            if "fake" in path:
                fake_vids.append(path)
                labels_fake.append(0)

        videos = real_vids + fake_vids
        labels = labels_real + labels_fake
        test_df = pd.DataFrame({'video': videos, 'label':labels})

        if len(test_df)<2:
        real_vids = []
        fake_vids = []
        labels_real = []
        labels_fake = []
        for path in glob.glob(root+"/**/*.avi", recursive=True):
            if "real" in path:
                real_vids.append(path)
                labels_real.append(1)
            if "fake" in path:
                fake_vids.append(path)
                labels_fake.append(0)

        videos = real_vids + fake_vids
        labels = labels_real + labels_fake
        test_df = pd.DataFrame({'video': videos, 'label':labels})
        return test_df




def run(args):
    inference(model=args.model, test_df=args.dataset,  normalization=args.norm, num_frames=args.frames)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,  required=True, help="Provide the path of model used to test.")
    parser.add_argument('--dataset', type=str,  required=True, help="Provide the name of the  dataset used to test.")
    parser.add_argument('--norm', type=str, default="imagenet", choices=['imagenet', 'xception'],\
         required=False, help="Provide the name of the  dataset used to test.")
    parser.add_argument('--frames', type=int,  required=True, help="Number of frames per video to test.")


    return args



if __name__ == "__main__":
    args = parse_arguments()
    run(args)
