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

def checkLayerCompatibility(model, device, verbose):
    global ie
    net = ie.read_network(model+'.xml', model+'.bin')
    supported_layers = ie.query_network(net, device)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers)>0:
        if verbose:
            print(' '*2+'{} - Not supported layers:'.format(device))
            for layer in not_supported_layers:
                dispLayerInfo(net.layers[layer])
        else:
            print(' '*2+'{} - Not supported layers:{}'.format(device, not_supported_layers))
    else:
        print('  {} - All layers are supported'.format(device))

def checkLayerCompatibilityForAllAvailableDevices(model, verbose):
    global ie
    for device in ie.available_devices:
        #print('Checking layer comatibility for {}'.format(device))
        checkLayerCompatibility(model, device, verbose)

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

def checkModel(model, compatibility, verbose):
    print('Model: {}'.format(model))
    print('IR version:{}'.format(getIRVersion(model)))
    if compatibility:
        checkLayerCompatibilityForAllAvailableDevices(model, verbose)
    checkInputOutput(model)
    print()

def dumpModel(model):
    global ie
    print('Model: {}'.format(model))
    print('IR version:{}'.format(getIRVersion(model)))
    net = ie.read_network(model+'.xml', model+'.bin')
    for layer in net.layers.keys():
        l = net.layers[layer]
        dispLayerInfo(l)
    checkInputOutput(model)
    print()

def grepLayers(model, regex, verbose, case_sensitive):
    global ie
    print('Model: {}'.format(model))
    print('IR version:{}'.format(getIRVersion(model)))
    net = ie.read_network(model+'.xml', model+'.bin')
    flag = 0 if case_sensitive else re.I
    for layer in net.layers.keys():
        l = net.layers[layer]
        if re.search(regex, layer, flag) != None:
            if verbose:
                dispLayerInfo(l)
            else:
                print(' '*2+'Found {}'.format(layer))
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input IR model path')
    parser.add_argument('-d', '--dir', type=str, help='input IR model directory')
    parser.add_argument('-c', '--compatibility', action='store_true', default=False, help='check layer compatibility')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='output detailed information')
    parser.add_argument('-f', '--full_dump', action='store_true', default=False, help='display detailed information of all layers in the models')
    parser.add_argument('-g', '--grep', type=str, help='search layers which matches to the given regex string')
    parser.add_argument('--case_sensitive', action='store_true', default=False, help='case sensitive on layer search')
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
        if args.grep != None:
            grepLayers(model, args.grep, args.verbose, args.case_sensitive)
        elif args.full_dump:
            dumpModel(model)
        else:
            checkModel(model, args.compatibility, args.verbose)

if __name__ == "__main__":
    main()
