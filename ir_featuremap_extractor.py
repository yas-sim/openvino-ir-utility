import os
import sys
import argparse
import copy
import pickle

import cv2
import xml.etree.ElementTree as et

from openvino.inference_engine import IECore

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


def findNodeFromXML(xmltree, nodeid):
    root = xmltree.getroot()
    layers = root.find('layers')
    for layer in layers.findall('layer'):
        if int(layer.attrib['id']) == nodeid:
            return layer
    return None


def modifyXMLForFeatureVectorProbing(xmltree, nodeid):
    xmlcopy = copy.deepcopy(xmltree)
    layer = findNodeFromXML(xmlcopy, nodeid)

    # obtain output port information of the target node (port # and dims)
    outport      = layer.find('output').find('port')
    outport_id   = int(outport.attrib['id'])
    outport_prec = outport.attrib['precision']
    outport_dims = outport.findall('dim')
    outport_dims_string = ""
    for dim in outport_dims:
        outport_dims_string += et.tostring(dim).decode('utf-8')

    # generate XML strings
    dummyLayer = """
    		<layer id="9999" name="featuremap_checker_dummy_node" type="Result" version="opset1">
			<input>
				<port id="0">
                {}
				</port>
			</input>
		</layer>
    """.format(outport_dims_string)
    dummyEdge = '		<edge from-layer="{}" from-port="{}" to-layer="9999" to-port="0"/>'.format(nodeid, outport_id)

    # modify XML to make a dummy branch path for feature map extraction
    xmlcopy.find('layers').append(et.fromstring(dummyLayer))
    xmlcopy.find('edges').append(et.fromstring(dummyEdge))

    # return the modified XML and the name of the target node (specified by 'nodeid')
    return xmlcopy, layer.attrib['name']



# TODO: You need to modify this function to make this fit to your model
#       E.g. - If your model uses multiple inputs, you need to prepare input data for those inputs
#            - If your model requires non-image data, you need to implement appropiate data preparation code and preprocessing for it
def prepareInputs(net_inputs, args):
    input_data = {}

    input_blob_names  = list(net_inputs.keys())

    input0Name = input_blob_names[0]
    input0Info = net_inputs[input0Name]
    N,C,H,W = input0Info.tensor_desc.dims
    img = cv2.imread(args.input)    # default = image.jpg
    img = cv2.resize(img, (W, H))
    img = img.transpose((2, 0, 1))
    img = img.reshape((1, C, H, W))
    input_data[input0Name] = img

    return input_data




def main(args):
    originalXML = readXML(args.model)
    weight = readBIN(args.model)

    print('node# : nodeName')
    feature_vectors = {}
    ie = IECore()
    root = originalXML.getroot()
    layers = root.find('layers')
    for layer in layers.findall('layer'):
        nodeid = int(layer.attrib['id'])
        nodetype = layer.attrib['type']
        if nodetype in ['Const']: # , 'ShapeOf', 'Convert', 'StridedSlice', 'PriorBox']:
            continue
        if not layer.find('output') is None:
            nodeName = layer.attrib['name']
            outputport = layer.find('output').find('port')
            proc = outputport.attrib['precision']
            dims = []
            for dim in outputport.findall('dim'):                       # extract shape information
                dims.append(dim.text)

            modifiedXML, targetNodeName = modifyXMLForFeatureVectorProbing(originalXML, nodeid)
            XMLstr = et.tostring(modifiedXML.getroot())
            print('{} : {}'.format(nodeid, targetNodeName))

            net = ie.read_network(XMLstr, weight, init_from_buffer=True)
            try:
                exenet = ie.load_network(net, 'CPU')
            except RuntimeError:
                #et.dump(modifiedXML)
                print('*** RuntimeError: load_network() -- Skip node \'{}\' - \'{}\''.format(targetNodeName, nodetype))
                continue

            inputs = prepareInputs(net.input_info, args)    # ToDo: Prepare inupts for inference. User may need to modify this function to generate appropriate input for the specific model.
            res = exenet.infer(inputs)[nodeName]

            feature_vectors[nodeName] = [proc, dims, res]
            #print(nodeName, res)
            del exenet
            del net

    dirname, filename = os.path.split(args.model)
    basename, extname = os.path.splitext(filename)
    fname = basename+'_featmap.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(feature_vectors, f)

    print('\nFeature maps are output to \'{}\''.format(fname))

if __name__ == "__main__":
    print('*** OpenVINO feature map extractor')
    print('@@@ This program takes \'image.jpg\' and supply to the 1st input blob as default.')
    print('@@@ In case your model requires special data input, you need to modify \'prepareInputs()\' function to meet the requirements.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input IR model path')
    parser.add_argument('-i', '--input', type=str, default='image.jpg', help='input image data path (default=image.jpg')
    args = parser.parse_args()

    main(args)
