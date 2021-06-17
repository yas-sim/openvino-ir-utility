import os
import sys
import argparse
import pickle
import struct

import xml.etree.ElementTree as et

from openvino.inference_engine import IECore

def calc_oblob_size(model):
    # for unpacking binary buffer
    format_config = { 'FP32': ['f', 4], 
                      'FP16': ['e', 2],
                      'I64' : ['q', 8],
                      'I32' : ['i', 4],
                      'I16' : ['h', 2],
                      'I8'  : ['b', 1],
                      'U8'  : ['B', 1]}

    print('layer name, layer type, port num, precision, oblob_size, [dims...]')
    # Parse IR XML file
    tree = et.parse(model+'.xml')
    root = tree.getroot()
    layers = root.find('layers')
    weight = {}
    oblob_size_total = 0
    for layer in layers:
        layer_name = layer.attrib['name']
        layer_type = layer.attrib['type']
        outputs = layer.find('output')
        if not outputs is None:
            ports = outputs.find('port')
            for oport in outputs:
                port_num  = oport.attrib['id']
                precision = oport.attrib['precision']
                prec_type = format_config[precision][0]
                prec_size = format_config[precision][1]
                print('{},{},{},{}'.format(layer_name, layer_type, port_num, precision), end='', flush=True)
                oblob_size = 1
                oblob_str = ''
                for dim in oport.findall('dim'):                       # extract shape information
                    oblob_str += ',{}'.format(dim.text)
                    oblob_size *= int(dim.text)
                oblob_size *= prec_size
                print(',{}{}'.format(oblob_size, oblob_str))
                oblob_size_total += oblob_size
    print('Total output blob size : {:,}B / {:,.2f}MB'.format(oblob_size_total, oblob_size_total/(1024*1024)))

def main():
    print('*** OpenVINO IR model output blob size calculator')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input IR model path')
    args = parser.parse_args()

    model, ext = os.path.splitext(args.model)
    if ext != '.xml':
        print('The specified model is not \'.xml\' file')
        sys.exit(-1)
    calc_oblob_size(model)

if __name__ == "__main__":
    main()
