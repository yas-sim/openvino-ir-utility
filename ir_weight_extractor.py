import os
import sys
import argparse
import pickle
import struct

import xml.etree.ElementTree as et

from openvino.inference_engine import IECore

def dumpWeight(model):
    # for unpacking binary buffer
    format_config = { 'FP32': ['f', 4], 
                      'FP16': ['e', 2],
                      'I64' : ['q', 8],
                      'I32' : ['i', 4],
                      'I16' : ['h', 2],
                      'I8'  : ['b', 1],
                      'U8'  : ['B', 1]}

    # Read IR weight data
    with open(model+'.bin', 'rb') as f:
        binWeight = f.read()

    # Parse IR XML file, find 'Const' node, extract weight, and generate pickle file 
    tree = et.parse(model+'.xml')
    root = tree.getroot()
    layers = root.find('layers')
    weight = {}
    print('    size : nodeName')
    for layer in layers:
        if layer.attrib['type'] == 'Const':
            data = layer.find('data')
            if not data is None:
                if 'offset' in data.attrib and 'size' in data.attrib:
                    offset = int(data.attrib['offset'])
                    size   = int(data.attrib['size'])
                    blobBin = binWeight[offset:offset+size]                     # cut out the weight for this blob from the weight buffer
                    outputport = layer.find('output').find('port')
                    prec = outputport.attrib['precision']
                    dims = []
                    for dim in outputport.findall('dim'):                       # extract shape information
                        dims.append(dim.text)
                    formatstring = '<' + format_config[prec][0] * (len(blobBin)//format_config[prec][1])
                    decodedwgt = struct.unpack(formatstring, blobBin)           # decode the buffer
                    weight[layer.attrib['name']] = [ prec, dims, decodedwgt ]         # { blobName : [ precStr, dims, weightBuf ]}
                    print('{:8} : {}'.format(len(blobBin), layer.attrib['name']))
    fname = model+'_wgt.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(weight, f)
    print('\n' + fname + ' is generated')

def main():
    print('*** OpenVINO IR model weight data extractor')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input IR model path')
    args = parser.parse_args()

    model, ext = os.path.splitext(args.model)
    if ext != '.xml':
        print('The specified model is not \'.xml\' file')
        sys.exit(-1)
    dumpWeight(model)

if __name__ == "__main__":
    main()
