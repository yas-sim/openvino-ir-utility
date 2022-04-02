import os
import sys
import argparse
import copy
import pickle

import cv2

from openvino.preprocess import PrePostProcessor
from openvino.runtime import AsyncInferQueue, Core, InferRequest, Layout, Type

type_table = { Type.f16: 'f16', Type.f32: 'f32', Type.f64: 'f64', Type.bf16: 'bf16',
               Type.i16: 'i16', Type.i32: 'i32', Type.i64: 'i64', Type.i8  : 'i8',
               Type.u16: 'u16', Type.u32: 'u32', Type.u64: 'u64', Type.u8  : 'u8',
               Type.bitwidth: 'bit', Type.boolean: 'bool' }


def splitFileName(file):
    dirname, filename = os.path.split(file)
    basename, extname = os.path.splitext(filename)
    return dirname, basename, extname

def readXML(model):
    dname, bname, ename = splitFileName(model)
    tree = et.parse(os.path.join(dname, bname+'.xml'))
    return tree

def readBIN(model):
    dname, bname, ename = splitFileName(model)
    with open(os.path.join(dname, bname+'.bin'), 'rb') as f:
        weight = f.read()
    return weight


# TODO: You need to modify this function to make this fit to your model
#       E.g. - If your model uses multiple inputs, you need to prepare input data for those inputs
#            - If your model requires non-image data, you need to implement appropiate data preparation code and preprocessing for it
def prepareInputs(ir_model, args):
    input_data = {}

    input0Name = ir_model.input().get_any_name()
    N,C,H,W = ir_model.input().shape
    img = cv2.imread(args.input)    # default = image.jpg
    img = cv2.resize(img, (W, H))
    img = img.transpose((2, 0, 1))
    img = img.reshape((1, C, H, W))
    input_data[input0Name] = img

    return input_data




def main(args):
    print('node# : nodeName')
    feature_vectors = {}
    ie = Core()

    ir_model = ie.read_model(args.model)
    ops = ir_model.get_ops()
    del ir_model

    for op_idx, op in enumerate(ops):
        op_name  = op.get_friendly_name()
        op_type  = op.type_info.name
        if op_type in [ 'Constant', 'Parameter', 'Result' ]:
            continue

        op_outputs = op.outputs()
        for out_idx, out in enumerate(op_outputs):
            out_shape = out.get_shape()
            out_shape = list(out_shape) if len(out_shape)>0 else []
            out_elem_type = type_table[out.get_element_type()]

            print(op_idx, op_name, op_type, out_idx, out_shape, out_elem_type)

            ir_model = ie.read_model(args.model)
            ir_model.add_outputs([(op_name, out_idx)])     # Add probe to snoop the feature map
            compiled_model = ie.compile_model(ir_model, args.device)

            inputs = prepareInputs(ir_model, args)         # ToDo: Prepare inupts for inference. User may need to modify this function to generate appropriate input for the specific model.
            res = compiled_model.infer_new_request(inputs)
            keys = list(res.keys())
            if len(keys)>1:
                #print(res[keys[1]])
                feature_vectors[op_name] = [out_idx, out_elem_type, out_shape, list(res[keys[1]])]
            del compiled_model
            del ir_model

    dirname, filename = os.path.split(args.model)
    basename, extname = os.path.splitext(filename)
    fname = basename+'_featmap.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(feature_vectors, f)

    print('\nFeature maps are output to \'{}\''.format(fname))

if __name__ == "__main__":
    print('*** OpenVINO feature map extractor')
    print('@@@ This program takes \'image.jpg\' and supply to the 1st input blob as default.')
    print('@@@ In case your model requires special input data, you need to modify \'prepareInputs()\' function to meet the requirements.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input IR model path')
    parser.add_argument('-i', '--input', type=str, default='image.jpg', help='input image data path (default=image.jpg)')
    parser.add_argument('-d', '--device', type=str, default='CPU', help='device to use')
    args = parser.parse_args()

    main(args)
