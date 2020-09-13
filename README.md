# Intel(r) OpenVINO(tm) IR model utility tool
This is a utility tool for OpenVINO IR model file. This tool has following functions:

1. Display IR model summary information (`ir-summary.py`)  
  Displays IR version, Input/Output blob names and shapes

2. Check layer compatibility (`ir-summary.py`)  
  Checks whether the layers are supported on the supported devices or not

3. Search layers (`ir-summary.py`)  
  Search layer from the IR models with regular expression

4. Dump layers (`ir-summary.py`)  
  Display all layers in the models (edge information will bot be displayed)

5. **[NEW]** Weight data extraction (`ir_weight_extractor.py`)  
  Extract weight data from an IR model and ganerates a pickle file

6. **[NEW]** Extract the feature map (intermediate buffer between layers) from an IR model (`ir_featuremap_extractor.py`)  
  Run an IR model and extract the feature map data of all layers and generates a pickle file

OpenVINOのIRモデル用のユーティリティーツールプログラムです。このツールは下記の機能を持っています。
1. IRモデルのサマリー情報表示 (`ir-summary.py`)  
  IRバージョン、input / output blob名、shape
2. レイヤー互換性情報 (`ir-summary.py`)  
  モデル内のどのレイヤーがどのデバイスでサポートできないかのチェック
3. レイヤー検索 (`ir-summary.py`)  
  正規表現を使ってレイヤーの検索
4. レイヤー情報のダンプ (`ir-summary.py`)  
  モデル内の全てのレイヤーを表示します（エッジの情報は表示しません）
5. **[NEW]** IRモデルの重みデータ抜き出し  (`ir_weight_extractor.py`)  
　重みデータを抜き出し、Python pickleファイルを作成します
6. **[NEW]** 特徴マップデータ抜き取り (`ir_featuremap_extractor.py`)  
　IRモデルを実行しながら中間バッファから特徴マップデータを抜き出し、Python pickleファイルを作成します


## How to Run

All tool are Python scripts. Just run them.

1. `ir-summary.py`  
*Options:*  
    `-m`, `--model` : Input IR model path  
    `-d`, `--dir`   : Input IR model directory  
    `-c`, `--compatibility` : Check layer compatibility  
    `-v`, `--verbose` : Output detailed information  
    `-f`, `--full_dump` : Display detailed information of all layers in the models  
    `-g`, `--grep` : Search layers which matches to the given regex string')  
    `--case_sensitive` : Case sensitive on layer search  

2. `ir_weight_extractor.py`  
  `model_wgt.pickle` file will be generated *at the same directory as the input model* when the input model file name is `model.xml`.  
*Options:*  
    `-m`, `--model` : Input IR model path  
*Output pickle file format:*  
Dictionary `{ blobName0 : [ precStr0, shape0, weightBuf0 ], blobName1 : [ precStr1, shape1, weightBuf1 ], ... }`  precStr='FP32', 'FP16', 'I32', ...

3. `ir_featuremap_extractor.py`  
  `model_featmap.pickle` file will be generated at the same directory as the script file when the input model file name is `model.xml`.  
*IMPORTANT NOTICE:*  
This program reads `image.jpg` and supply it to the 1st input blob as default. In case your model requires special input data (multiple input blobs, non-image input, etc), you need to modify `prepareInputs()` function to meet the requirements.  
*Options:*  
    `-m`, `--model` : Input IR model path  
    `-i`, `--input` : Input image file path (default=`image.jpg`)  
*Output pickle file format:*  
Dictionary `{ blobName0 : featmap0, blobName1 : featmap1, ... }`


## How to read and use Python pickled data
```python
import pickle
with open('foo.pickle', 'rb') as f:
    data = pickle.load(f)     # the pickled data will be loaded to `data`

a = data['conv1']
```

## Examples of command output

### Model summary - single input file (IR version, input / output blob name and shape)
```sh
python ir-summary.py -m ..\..\public\googlenet-v1\FP16\googlenet-v1.xml
Model: ..\..\public\googlenet-v1\FP16\googlenet-v1
IR version:10
Input Blob(s):
  BlobName:data, Shape:[1, 3, 224, 224], Precision:FP32
Output Blob(s):
  BlobName:prob, Shape:[1, 1000], Precision:FP32
```

### Model summary - multiple input files (IR version, input / output blob name and shape)
```sh
> python ir-summary.py -d ..\..\public
Model: ..\..\public\googlenet-v1\FP16\googlenet-v1
IR version:10
Input Blob(s):
  BlobName:data, Shape:[1, 3, 224, 224], Precision:FP32
Output Blob(s):
  BlobName:prob, Shape:[1, 1000], Precision:FP32

Model: ..\..\public\googlenet-v1\FP32\googlenet-v1
IR version:10
Input Blob(s):
  BlobName:data, Shape:[1, 3, 224, 224], Precision:FP32
Output Blob(s):
  BlobName:prob, Shape:[1, 1000], Precision:FP32
  :
```

### Search layers - simple mode
```sh
python ir-summary.py -m ..\..\public\googlenet-v1\FP16\googlenet-v1.xml -g conv
Model: ..\..\public\googlenet-v1\FP16\googlenet-v1
IR version:10
  Found conv1/7x7_s2
  Found conv1/relu_7x7
  Found conv2/3x3_reduce
  Found conv2/relu_3x3_reduce
  Found conv2/3x3
  Found conv2/relu_3x3
  Found conv2/norm26960
```

### Search layers - verbose mode
```sh
python ir-summary.py -m ..\..\public\googlenet-v1\FP16\googlenet-v1.xml -g conv1/7x7 -v
Model: ..\..\public\googlenet-v1\FP16\googlenet-v1
IR version:10
    LayerName:conv1/7x7_s2, param:{'dilations': '1,1', 'group': '1', 'kernel': '7,7', 'output': '64', 'pads_begin': '3,3', 'pads_end': '3,3', 'strides': '2,2'}, type:Convolution
      InputName:Add_, layout:NCHW, precision:FP16, shape:[1, 3, 224, 224]
      OutputName:conv1/7x7_s2, layout:NCHW, precision:FP16, shape:[1, 64, 112, 112]
```

### Layer compatibility check - simple mode
```sh
python ir-summary.py -m ..\..\public\googlenet-v1\FP16\googlenet-v1.xml -c
Model: ..\..\public\googlenet-v1\FP16\googlenet-v1
IR version:10
[E:] [BSL] found 0 ioexpander device
  CPU - All layers are supported
  GNA - Not supported layers:['pool1/norm16956', 'conv2/norm26960', 'prob']
  MYRIAD - Not supported layers:['data']
Input Blob(s):
  BlobName:data, Shape:[1, 3, 224, 224], Precision:FP32
Output Blob(s):
  BlobName:prob, Shape:[1, 1000], Precision:FP32
```

### Layer compatibility check - verbose mode
```sh
python ir-summary.py -m ..\..\public\googlenet-v1\FP16\googlenet-v1.xml -c -v
Model: ..\..\public\googlenet-v1\FP16\googlenet-v1
IR version:10
[E:] [BSL] found 0 ioexpander device
  CPU - All layers are supported
  GNA - Not supported layers:
    LayerName:pool1/norm16956, param:{'alpha': '0.000099999997474', 'beta': '0.75', 'k': '1', 'local-size': '5', 'region': 'across'}, type:Norm
      InputName:pool1/3x3_s2, layout:NCHW, precision:FP16, shape:[1, 64, 56, 56]
      OutputName:pool1/norm16956, layout:NCHW, precision:FP16, shape:[1, 64, 56, 56]
    LayerName:conv2/norm26960, param:{'alpha': '0.000099999997474', 'beta': '0.75', 'k': '1', 'local-size': '5', 'region': 'across'}, type:Norm
      InputName:conv2/relu_3x3, layout:NCHW, precision:FP16, shape:[1, 192, 56, 56]
      OutputName:conv2/norm26960, layout:NCHW, precision:FP16, shape:[1, 192, 56, 56]
    LayerName:prob, param:{'axis': '1'}, type:SoftMax
      InputName:loss3/classifier, layout:NC, precision:FP16, shape:[1, 1000]
      OutputName:prob, layout:NC, precision:FP32, shape:[1, 1000]
  MYRIAD - Not supported layers:
    LayerName:data, param:{}, type:Input
      OutputName:data, layout:NCHW, precision:FP32, shape:[1, 3, 224, 224]
Input Blob(s):
  BlobName:data, Shape:[1, 3, 224, 224], Precision:FP32
Output Blob(s):
  BlobName:prob, Shape:[1, 1000], Precision:FP32
```

### Model full dump
```sh
python ir-summary.py -m ..\..\public\googlenet-v1\FP16\googlenet-v1.xml -f
Model: ..\..\public\googlenet-v1\FP16\googlenet-v1
IR version:10
    LayerName:data, param:{}, type:Input
      OutputName:data, layout:NCHW, precision:FP32, shape:[1, 3, 224, 224]
    LayerName:Add_, param:{}, type:ScaleShift
      InputName:data, layout:NCHW, precision:FP32, shape:[1, 3, 224, 224]
      OutputName:Add_, layout:NCHW, precision:FP16, shape:[1, 3, 224, 224]
    LayerName:conv1/7x7_s2, param:{'dilations': '1,1', 'group': '1', 'kernel': '7,7', 'output': '64', 'pads_begin': '3,3', 'pads_end': '3,3', 'strides': '2,2'}, type:Convolution
      InputName:Add_, layout:NCHW, precision:FP16, shape:[1, 3, 224, 224]
      OutputName:conv1/7x7_s2, layout:NCHW, precision:FP16, shape:[1, 64, 112, 112]
    LayerName:conv1/relu_7x7, param:{}, type:ReLU
      InputName:conv1/7x7_s2, layout:NCHW, precision:FP16, shape:[1, 64, 112, 112]
      OutputName:conv1/relu_7x7, layout:NCHW, precision:FP16, shape:[1, 64, 112, 112]
    :
```

### IR model weight extraction - `ir_weight_extractor.py`
```sh
python ir_weight_extractor.py -m public\googlenet-v1\FP16\googlenet-v1.xml
*** OpenVINO IR model weight data extractor
size : nodeName
6 data_add_/copy_const
18816 175/Output_0/Data__const
128 conv1/7x7_s2/Dims2528/copy_const
8 pool1/norm16956/value6958_const
  :
256 inception_5b/pool_proj/Dims2558/copy_const
16 loss3/classifier/flatten_fc_input/Cast_18765_const
2048000 loss3/classifier/WithoutBiases/1_port_transpose6482_const
2000 238/Output_0/Data_/copy_const

public\googlenet-v1\FP16\googlenet-v1_wgt.pickle is generated
```

### Feature map extraction - `ir_featuremap_extractor.py`
```sh
python ir_featuremap_checker.py -m googlenet-v1.xml -i car_1.bmp
*** OpenVINO feature map extractor
@@@ This program takes 'image.jpg' and supply to the 1st input blob as default.
@@@ In case your model requires special data input, you need to modify 'prepareInputs()' function to meet the requirements.
node# : nodeName
0 : data
2 : Add_
4 : conv1/7x7_s2/WithoutBiases
6 : conv1/7x7_s2
 :      :
318 : loss3/classifier/WithoutBiases
320 : loss3/classifier
321 : prob

Feature maps are output to 'googlenet-v1_featmap.pickle'
```

## Test environment
- Windows 10
- Intel(r) Distribution of OpenVINO(tm) toolkit 2020.2
- Intel(r) Distribution of OpenVINO(tm) toolkit 2020.4