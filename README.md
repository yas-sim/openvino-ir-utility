# Intel(r) OpenVINO(tm) IR model utility tool
This is a utility tool for OpenVINO IR model file. This tool has following functions:
1. Display IR model summary information  
  Displays IR version, Input/Output blob names and shapes
2. Check layer compatibility  
  Checks whether the layers are supported on the supported devices or not
3. Search layers  
  Search layer from the IR models with regular expression
4. Dump layers  
  Display all layers in the models (edge information will bot be displayed)

OpenVINOのIRモデル用のユーティリティーツールプログラムです。このツールは下記の機能を持っています。
1. IRモデルのサマリー情報表示  
  IRバージョン、input / output blob名、shape
2. レイヤー互換性情報  
  モデル内のどのレイヤーがどのデバイスでサポートできないかのチェック
3. レイヤー検索  
  正規表現を使ってレイヤーの検索
4. レイヤー情報のダンプ  
  モデル内の全てのレイヤーを表示します（エッジの情報は表示しません）

## How to Run

All tool are Python scripts. Just run them.

*Options:*  
    `-m`, `--model` : Input IR model path  
    `-d`, `--dir`   : Input IR model directory  
    `-c`, `--compatibility` : Check layer compatibility  
    `-v`, `--verbose` : Output detailed information  
    `-f`, `--full_dump` : Display detailed information of all layers in the models  
    `-g`, `--grep` : Search layers which matches to the given regex string')  
    `--case_sensitive` : Case sensitive on layer search  

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

## Test environment
- Windows 10
- Intel(r) Distribution of OpenVINO(tm) toolkit 2020.2
