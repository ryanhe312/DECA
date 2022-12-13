# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device

    # load test images 
    print(args.image_path)
    testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector, scale=1)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector, scale=1)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    # identity reference
    i = 0
    name = testdata[i]['imagename']
    os.makedirs(os.path.join(savefolder,name), exist_ok=True)
    images = testdata[i]['image'].to(device)[None,...]
    with torch.no_grad():
        id_codedict = deca.encode(images)
    id_opdict, id_visdict = deca.decode(id_codedict)
    id_visdict = {x:id_visdict[x] for x in ['inputs', 'shape_detail_images']}   

    # -- expression transfer
    # exp code from image
    exp_name = expdata[i]['imagename']
    exp_images = expdata[i]['image'].to(device)[None,...]
    with torch.no_grad():
        exp_codedict = deca.encode(exp_images)
    # transfer exp code

    id_codedict_copy = id_codedict.copy()
    asym_face = 0
    for i in range(11):
        id_codedict = id_codedict_copy.copy()
        # id_codedict['pose'][:,3:] = id_codedict['pose'][:,3:] * (1-i/10) + exp_codedict['pose'][:,3:] * (i/10)
        id_codedict['exp'] = id_codedict['exp'] * (1-i/10) + exp_codedict['exp'] * (i/10)
        if i == 0:
            transfer_opdict, transfer_visdict = deca.decode(id_codedict)
            origin_vert = transfer_opdict['origin_vert']
        else:
            transfer_opdict, transfer_visdict = deca.decode(id_codedict,origin_vert=origin_vert, asym_face=asym_face)
        id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
        # cv2.imwrite(os.path.join(savefolder, name + '_animation.jpg'), deca.visualize(id_visdict))

        transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name, 'reconstruction'), exist_ok=True)
            os.makedirs(os.path.join(savefolder, name, 'animation'), exist_ok=True)
        
        # -- save results
        visdict = transfer_visdict; opdict = transfer_opdict
        for vis_name in ['rendered_images']:
            if vis_name not in visdict.keys():
                continue
            image  =util.tensor2image(visdict[vis_name][0])
            cv2.imwrite(os.path.join(savefolder, name, name + '_' + exp_name + '_level_' + str(i) +'.png'), util.tensor2image(visdict[vis_name][0]))

        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '_' + exp_name + '_level_' + str(i) + '.obj'), transfer_opdict) 

        if args.saveMat:
            numpy_dict = dict(shape=id_codedict['shape'].cpu().numpy(),exp=id_codedict['exp'].cpu().numpy(),pose=id_codedict['pose'].cpu().numpy())
            np.save(os.path.join(savefolder, name, name + '_' + exp_name + '_level_' + str(i)),numpy_dict) 

        print(f'-- please check the results in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--image_path', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp/8.jpg', type=str, 
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details' )
    # save
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())
