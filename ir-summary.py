import os
import sys
import glob
import argparse
import re

import xml.etree.ElementTree as et

from openvino.inference_engine import IECore

ie = IECore()

def dispLayerInfo(layer, indent=4):
    l = layer
    print(' '*indent+'LayerName:{}, param:{}, type:{}'.format(layer.name, layer.params, layer.type))
    for i in layer.in_data:
        print(' '*(indent+2)+'InputName:{}, layout:{}, precision:{}, shape:{}'.format(i.name, i.layout, i.precision, i.shape))
    for o in layer.out_data:
        print(' '*(indent+2)+'OutputName:{}, layout:{}, precision:{}, shape:{}'.format(o.name, o.layout, o.precision, o.shape))

def checkInputOutput(model):
    global ie
    net = ie.read_network(model+'.xml', model+'.bin')
    print('Input Blob(s):')
    for input in net.input_info:
        print('  BlobName:{}, Shape:{}, Precision:{}'.format(input, net.input_info[input].tensor_desc.dims, net.input_info[input].precision))
    print('Output Blob(s):')
    for output in net.outputs:
        print('  BlobName:{}, Shape:{}, Precision:{}'.format(output, net.outputs[output].shape, net.outputs[output].precision))

def getIRVersion(model):
    tree = et.parse(model+'.xml')
    root = tree.getroot()
    ver = int(root.attrib['version'])
    return ver

def checkModel(model, verbose):
    print('Model: {}'.format(model))
    print('IR version:{}'.format(getIRVersion(model)))
    checkInputOutput(model)
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input IR model path')
    parser.add_argument('-d', '--dir', type=str, help='input IR model directory')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='output detailed information')
    args = parser.parse_args()

    models = []
    if args.model != None:
        model, ext = os.path.splitext(args.model)
        if ext != '.xml':
            print('The specified model is not \'.xml\' file')
            sys.exit(-1)
        if os.path.exists(model+'.bin'):
            models.append(model)
    elif args.dir != None:
        xmls = glob.glob(os.path.join(args.dir, '**', '*.xml'), recursive=True)
        for xml in xmls:
            base, ext = os.path.splitext(xml)
            if os.path.exists(base+'.bin'):
                models.append(base)
    else:
        parser.print_usage()

    for model in models:
        checkModel(model, args.verbose)

if __name__ == "__main__":
    main()
