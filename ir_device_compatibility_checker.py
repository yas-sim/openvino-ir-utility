import os
import sys
import argparse

import xml.etree.ElementTree as et

def readDeviceCompatibilityList(filename):
    with open(filename, 'rt') as f:
        compList = {}
        label = None
        for ln in f:
            if ln[0] == '#':
                continue
            ln = ln.strip().replace('*', '').split('\t')
            if label is None:     # The 1st line == label
                label = ln
                continue
            layer = ''
            for dev, sts in zip(label, ln):
                if dev == 'LAYERS':
                    layer = sts
                    compList[layer] = {}
                    continue
                else:
                    sts = True if sts == 'Supported' else False
                compList[layer][dev] = sts
    return compList

def checkCompatibility(model, compListFile):
    compList = readDeviceCompatibilityList(compListFile)

    # Parse IR XML file, find 'Const' node, extract weight, and generate pickle file 
    tree = et.parse(model+'.xml')
    root = tree.getroot()
    layers = root.find('layers')
    device = 'CPU'
    #print(compList)
    results = { 'supported':[], 'unsupported':[], 'unknown':[] }
    for layer in layers:
        layerName = layer.attrib['name']
        layerType = layer.attrib['type']
        #print(layerName, layerType)
        if layerType in compList:
            sts = compList[layerType][device]
            if sts:
                result = 'supported'
            else:
                result = 'unsupported'
        else:
            result = 'unknown'
        results[result].append([layerName, layerType])
    return results

def printResults(results):
    for sts in results:
        print(sts, '-'*40)
        for item in results[sts]:
            print(item[1],',', end='', flush=True)
        print()

def main():
    print('*** OpenVINO IR model weight data extractor')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input IR model path')
    parser.add_argument('-l', '--list', type=str, default='supported_layers.txt', help='Layer and device comatibility table text file')
    args = parser.parse_args()

    model, ext = os.path.splitext(args.model)
    if ext != '.xml':
        print('The specified model is not \'.xml\' file')
        sys.exit(-1)

    results = checkCompatibility(model, args.list)
    #printResults(results)

if __name__ == "__main__":
    main()
