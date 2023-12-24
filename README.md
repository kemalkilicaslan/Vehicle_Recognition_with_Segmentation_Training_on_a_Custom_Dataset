# Vehicle Recognition with Segmentation Training on a Custom Dataset



In this project, firstly, 20 randomly selected images of cars, 20 pickups and 20 trucks were segmented using Roboflow, then these images were classified as 70% train, 20% validation, 10% test, and in the preprocessing stage, auto-oriented and resized to 640x640, data augmentation stage,
**Flip**:Horizontal and Vertical, **90°Rotate**: Clockwise, Counter-Clockwise, Upside Down, **Rotation**: Between -15° and +15°, **Grayscale**: 100% applied, **Blur**: 2.75px to train the model by applying the operations.

In the last stage, we will measure the performance of the model that recognises 15 images consisting of 5 cars, 5 pickups and 5 trucks that are not included in the trained model with a high rate of prediction.

### Install YOLOv8


```python
!pip install ultralytics
```

    Collecting ultralytics
      Downloading ultralytics-8.0.222-py3-none-any.whl (653 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m654.0/654.0 kB[0m [31m12.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (3.7.1)
    Requirement already satisfied: numpy>=1.22.2 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.23.5)
    Requirement already satisfied: opencv-python>=4.6.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.8.0.76)
    Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.4.0)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (6.0.1)
    Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.31.0)
    Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.11.4)
    Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.1.0+cu118)
    Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.16.0+cu118)
    Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.66.1)
    Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.5.3)
    Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.12.2)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from ultralytics) (5.9.5)
    Requirement already satisfied: py-cpuinfo in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.0.0)
    Collecting thop>=0.1.1 (from ultralytics)
      Downloading thop-0.1.1.post2209072238-py3-none-any.whl (15 kB)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (4.45.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.4.5)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (23.2)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->ultralytics) (2023.3.post1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2023.11.17)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (4.5.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.1.2)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2023.6.0)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2.1.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.8.0->ultralytics) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.8.0->ultralytics) (1.3.0)
    Installing collected packages: thop, ultralytics
    Successfully installed thop-0.1.1.post2209072238 ultralytics-8.0.222



```python
from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
!yolo checks
```

    Ultralytics YOLOv8.0.222 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    Setup complete ✅ (2 CPUs, 12.7 GB RAM, 26.9/78.2 GB disk)
    
    OS                  Linux-5.15.120+-x86_64-with-glibc2.35
    Environment         Colab
    Python              3.10.12
    Install             pip
    RAM                 12.68 GB
    CPU                 Intel Xeon 2.00GHz
    CUDA                11.8
    
    matplotlib          ✅ 3.7.1>=3.3.0
    numpy               ✅ 1.23.5>=1.22.2
    opencv-python       ✅ 4.8.0.76>=4.6.0
    pillow              ✅ 9.4.0>=7.1.2
    pyyaml              ✅ 6.0.1>=5.3.1
    requests            ✅ 2.31.0>=2.23.0
    scipy               ✅ 1.11.4>=1.4.1
    torch               ✅ 2.1.0+cu118>=1.8.0
    torchvision         ✅ 0.16.0+cu118>=0.9.0
    tqdm                ✅ 4.66.1>=4.64.0
    pandas              ✅ 1.5.3>=1.1.4
    seaborn             ✅ 0.12.2>=0.11.0
    psutil              ✅ 5.9.5
    py-cpuinfo          ✅ 9.0.0
    thop                ✅ 0.1.1-2209072238>=0.1.1


### YOLOv8 Model Trained on Custom Dataset


```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("iamkilicaslan").project("vehicle-segmentation-yvbo4")
dataset = project.version(1).download("yolov8")
```

    Collecting roboflow
      Downloading roboflow-1.1.11-py3-none-any.whl (68 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m68.5/68.5 kB[0m [31m1.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting certifi==2023.7.22 (from roboflow)
      Downloading certifi-2023.7.22-py3-none-any.whl (158 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m158.3/158.3 kB[0m [31m10.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting chardet==4.0.0 (from roboflow)
      Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m178.7/178.7 kB[0m [31m22.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cycler==0.10.0 (from roboflow)
      Downloading cycler-0.10.0-py2.py3-none-any.whl (6.5 kB)
    Collecting idna==2.10 (from roboflow)
      Downloading idna-2.10-py2.py3-none-any.whl (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.8/58.8 kB[0m [31m7.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.4.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from roboflow) (3.7.1)
    Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.23.5)
    Collecting opencv-python-headless==4.8.0.74 (from roboflow)
      Downloading opencv_python_headless-4.8.0.74-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.1/49.1 MB[0m [31m15.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from roboflow) (9.4.0)
    Collecting pyparsing==2.4.7 (from roboflow)
      Downloading pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.8/67.8 kB[0m [31m9.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.8.2)
    Collecting python-dotenv (from roboflow)
      Downloading python_dotenv-1.0.0-py3-none-any.whl (19 kB)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.31.0)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.16.0)
    Collecting supervision (from roboflow)
      Downloading supervision-0.16.0-py3-none-any.whl (72 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m72.2/72.2 kB[0m [31m10.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: urllib3>=1.26.6 in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.0.7)
    Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.66.1)
    Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (6.0.1)
    Collecting requests-toolbelt (from roboflow)
      Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.5/54.5 kB[0m [31m7.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting python-magic (from roboflow)
      Downloading python_magic-0.4.27-py2.py3-none-any.whl (13 kB)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (1.2.0)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (4.45.1)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (23.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->roboflow) (3.3.2)
    Requirement already satisfied: scipy<2.0.0,>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from supervision->roboflow) (1.11.4)
    Installing collected packages: python-magic, python-dotenv, pyparsing, opencv-python-headless, idna, cycler, chardet, certifi, supervision, requests-toolbelt, roboflow
      Attempting uninstall: pyparsing
        Found existing installation: pyparsing 3.1.1
        Uninstalling pyparsing-3.1.1:
          Successfully uninstalled pyparsing-3.1.1
      Attempting uninstall: opencv-python-headless
        Found existing installation: opencv-python-headless 4.8.1.78
        Uninstalling opencv-python-headless-4.8.1.78:
          Successfully uninstalled opencv-python-headless-4.8.1.78
      Attempting uninstall: idna
        Found existing installation: idna 3.6
        Uninstalling idna-3.6:
          Successfully uninstalled idna-3.6
      Attempting uninstall: cycler
        Found existing installation: cycler 0.12.1
        Uninstalling cycler-0.12.1:
          Successfully uninstalled cycler-0.12.1
      Attempting uninstall: chardet
        Found existing installation: chardet 5.2.0
        Uninstalling chardet-5.2.0:
          Successfully uninstalled chardet-5.2.0
      Attempting uninstall: certifi
        Found existing installation: certifi 2023.11.17
        Uninstalling certifi-2023.11.17:
          Successfully uninstalled certifi-2023.11.17
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    lida 0.0.10 requires fastapi, which is not installed.
    lida 0.0.10 requires kaleido, which is not installed.
    lida 0.0.10 requires python-multipart, which is not installed.
    lida 0.0.10 requires uvicorn, which is not installed.[0m[31m
    [0mSuccessfully installed certifi-2023.7.22 chardet-4.0.0 cycler-0.10.0 idna-2.10 opencv-python-headless-4.8.0.74 pyparsing-2.4.7 python-dotenv-1.0.0 python-magic-0.4.27 requests-toolbelt-1.0.0 roboflow-1.1.11 supervision-0.16.0




    loading Roboflow workspace...
    loading Roboflow project...
    Dependency ultralytics==8.0.196 is required but found version=8.0.222, to fix: `pip install ultralytics==8.0.196`


    Downloading Dataset Version Zip in vehicle-segmentation-1 to yolov8:: 100%|██████████| 7768/7768 [00:01<00:00, 5384.29it/s] 

    


    
    Extracting Dataset Version Zip to vehicle-segmentation-1 in yolov8:: 100%|██████████| 300/300 [00:00<00:00, 7213.32it/s]


### Load the YOLOv8x-seg model and train model



```python
!yolo task=segment mode=train model=yolov8x-seg.pt data={dataset.location}/data.yaml epochs=20 imgsz=640
```

    Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt to 'yolov8x-seg.pt'...
    100% 137M/137M [00:00<00:00, 252MB/s]
    Ultralytics YOLOv8.0.222 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    [34m[1mengine/trainer: [0mtask=segment, mode=train, model=yolov8x-seg.pt, data=/content/vehicle-segmentation-1/data.yaml, epochs=20, patience=50, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=runs/segment/train
    Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...
    100% 755k/755k [00:00<00:00, 90.2MB/s]
    2023-12-04 01:27:45.418173: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2023-12-04 01:27:45.418247: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2023-12-04 01:27:45.418288: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    Overriding model.yaml nc=80 with nc=3
    
                       from  n    params  module                                       arguments                     
      0                  -1  1      2320  ultralytics.nn.modules.conv.Conv             [3, 80, 3, 2]                 
      1                  -1  1    115520  ultralytics.nn.modules.conv.Conv             [80, 160, 3, 2]               
      2                  -1  3    436800  ultralytics.nn.modules.block.C2f             [160, 160, 3, True]           
      3                  -1  1    461440  ultralytics.nn.modules.conv.Conv             [160, 320, 3, 2]              
      4                  -1  6   3281920  ultralytics.nn.modules.block.C2f             [320, 320, 6, True]           
      5                  -1  1   1844480  ultralytics.nn.modules.conv.Conv             [320, 640, 3, 2]              
      6                  -1  6  13117440  ultralytics.nn.modules.block.C2f             [640, 640, 6, True]           
      7                  -1  1   3687680  ultralytics.nn.modules.conv.Conv             [640, 640, 3, 2]              
      8                  -1  3   6969600  ultralytics.nn.modules.block.C2f             [640, 640, 3, True]           
      9                  -1  1   1025920  ultralytics.nn.modules.block.SPPF            [640, 640, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  3   7379200  ultralytics.nn.modules.block.C2f             [1280, 640, 3]                
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  3   1948800  ultralytics.nn.modules.block.C2f             [960, 320, 3]                 
     16                  -1  1    922240  ultralytics.nn.modules.conv.Conv             [320, 320, 3, 2]              
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  3   7174400  ultralytics.nn.modules.block.C2f             [960, 640, 3]                 
     19                  -1  1   3687680  ultralytics.nn.modules.conv.Conv             [640, 640, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  3   7379200  ultralytics.nn.modules.block.C2f             [1280, 640, 3]                
     22        [15, 18, 21]  1  12319097  ultralytics.nn.modules.head.Segment          [3, 32, 320, [320, 640, 640]] 
    YOLOv8x-seg summary: 401 layers, 71753737 parameters, 71753721 gradients, 344.5 GFLOPs
    
    Transferred 651/657 items from pretrained weights
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/segment/train', view at http://localhost:6006/
    Freezing layer 'model.22.dfl.conv.weight'
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks with YOLOv8n...
    Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt to 'yolov8n.pt'...
    100% 6.23M/6.23M [00:00<00:00, 268MB/s]
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1mtrain: [0mScanning /content/vehicle-segmentation-1/train/labels... 126 images, 0 backgrounds, 0 corrupt: 100% 126/126 [00:00<00:00, 1116.27it/s]
    [34m[1mtrain: [0mNew cache created: /content/vehicle-segmentation-1/train/labels.cache
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    [34m[1mval: [0mScanning /content/vehicle-segmentation-1/valid/labels... 12 images, 0 backgrounds, 0 corrupt: 100% 12/12 [00:00<00:00, 872.24it/s]
    [34m[1mval: [0mNew cache created: /content/vehicle-segmentation-1/valid/labels.cache
    Plotting labels to runs/segment/train/labels.jpg... 
    [34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    [34m[1moptimizer:[0m AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 117 weight(decay=0.0005), 116 bias(decay=0.0)
    Image sizes 640 train, 640 val
    Using 2 dataloader workers
    Logging results to [1mruns/segment/train[0m
    Starting training for 20 epochs...
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           1/20      14.3G     0.6256      2.666      3.087      1.208         29        640: 100% 8/8 [00:14<00:00,  1.78s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.05it/s]
                       all         12         12      0.546      0.667      0.443      0.414      0.546      0.667      0.443      0.405
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           2/20      14.1G     0.6171      1.132       1.89      1.152         35        640: 100% 8/8 [00:11<00:00,  1.46s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.06it/s]
                       all         12         12      0.422      0.667      0.275       0.22      0.422      0.667      0.269        0.2
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           3/20      12.3G      0.663      1.165      1.646      1.199         33        640: 100% 8/8 [00:15<00:00,  1.95s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.81it/s]
                       all         12         12      0.333     0.0833   0.000395   0.000356      0.333     0.0833   0.000395   0.000237
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           4/20      12.4G     0.7647      1.172      1.555      1.251         38        640: 100% 8/8 [00:19<00:00,  2.41s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.97it/s]
                       all         12         12    0.00225        0.5    0.00384    0.00234    0.00285      0.667    0.00488    0.00304
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           5/20      12.3G     0.8437      1.149      1.363      1.278         42        640: 100% 8/8 [00:20<00:00,  2.53s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.92it/s]
                       all         12         12    0.00366        0.5    0.00304    0.00225    0.00366        0.5    0.00267     0.0018
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           6/20      12.2G      0.899      1.226      1.541      1.335         35        640: 100% 8/8 [00:19<00:00,  2.43s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.82it/s]
                       all         12         12      0.133      0.583      0.226      0.125      0.125        0.5      0.216      0.158
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           7/20      12.6G       1.01      1.577      1.536      1.398         40        640: 100% 8/8 [00:19<00:00,  2.44s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.99it/s]
                       all         12         12      0.144       0.75      0.231      0.116      0.122      0.667      0.214       0.13
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           8/20      13.4G     0.8598      1.162        1.4      1.307         38        640: 100% 8/8 [00:20<00:00,  2.51s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.80it/s]
                       all         12         12      0.113      0.667      0.133       0.07      0.108      0.583      0.128     0.0773
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
           9/20      12.6G     0.8787      1.303      1.314      1.354         40        640: 100% 8/8 [00:19<00:00,  2.44s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.03it/s]
                       all         12         12     0.0845      0.408      0.064     0.0332     0.0981      0.327     0.0329    0.00863
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          10/20      13.2G     0.8957      1.359      1.495      1.286         36        640: 100% 8/8 [00:19<00:00,  2.40s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.92it/s]
                       all         12         12     0.0845      0.408      0.064     0.0332     0.0981      0.327     0.0329    0.00863
    Closing dataloader mosaic
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          11/20      12.6G     0.8957      1.427      1.587      1.548         14        640: 100% 8/8 [00:20<00:00,  2.59s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.98it/s]
                       all         12         12     0.0845      0.408      0.064     0.0332     0.0981      0.327     0.0329    0.00863
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          12/20      12.7G     0.7397      1.164      1.412      1.415         14        640: 100% 8/8 [00:19<00:00,  2.43s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.87it/s]
                       all         12         12      0.103      0.333     0.0458     0.0368      0.103      0.333     0.0458     0.0368
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          13/20      13.2G     0.7775      1.234      1.303      1.435         14        640: 100% 8/8 [00:19<00:00,  2.41s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.05it/s]
                       all         12         12      0.226        0.5      0.236      0.106      0.123      0.333      0.158     0.0852
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          14/20      13.9G     0.7347      1.288      1.131      1.411         14        640: 100% 8/8 [00:19<00:00,  2.41s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.86it/s]
                       all         12         12      0.269      0.833      0.375      0.249      0.269      0.833      0.375       0.27
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          15/20      12.7G     0.5727       1.02     0.9376      1.294         14        640: 100% 8/8 [00:19<00:00,  2.45s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.06it/s]
                       all         12         12       0.29       0.75      0.475      0.301      0.272      0.667      0.449      0.353
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          16/20      13.7G     0.5724     0.8556     0.8862      1.228         14        640: 100% 8/8 [00:19<00:00,  2.41s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.06it/s]
                       all         12         12      0.138      0.583      0.334      0.246      0.138      0.583      0.355      0.216
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          17/20      13.8G     0.5616     0.9399     0.7815      1.279         14        640: 100% 8/8 [00:19<00:00,  2.40s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.07it/s]
                       all         12         12      0.475      0.657      0.411      0.264       0.49       0.74      0.534      0.327
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          18/20        13G     0.5294     0.7918     0.6991       1.23         14        640: 100% 8/8 [00:19<00:00,  2.41s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.01it/s]
                       all         12         12      0.575      0.665      0.569      0.454      0.575      0.665      0.569      0.431
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          19/20      13.1G     0.4133     0.6633     0.6426      1.114         14        640: 100% 8/8 [00:19<00:00,  2.41s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  2.03it/s]
                       all         12         12      0.785      0.705      0.833      0.724      0.762      0.609      0.787      0.687
    
          Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
          20/20      12.7G     0.4146     0.6559     0.5865      1.115         14        640: 100% 8/8 [00:19<00:00,  2.42s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.92it/s]
                       all         12         12      0.916      0.862      0.978      0.877      0.916      0.862      0.978      0.816
    
    20 epochs completed in 0.144 hours.
    Optimizer stripped from runs/segment/train/weights/last.pt, 143.9MB
    Optimizer stripped from runs/segment/train/weights/best.pt, 143.9MB
    
    Validating runs/segment/train/weights/best.pt...
    Ultralytics YOLOv8.0.222 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    YOLOv8x-seg summary (fused): 295 layers, 71723545 parameters, 0 gradients, 343.7 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  1.93it/s]
                       all         12         12      0.916      0.862      0.978      0.877      0.916      0.862      0.978      0.816
                       car         12          4      0.783          1      0.995      0.921      0.783          1      0.995      0.958
                    pickup         12          4      0.964          1      0.995      0.811      0.964          1      0.995      0.663
                     truck         12          4          1      0.586      0.945      0.899          1      0.586      0.945      0.828
    Speed: 0.2ms preprocess, 35.5ms inference, 0.0ms loss, 1.2ms postprocess per image
    Results saved to [1mruns/segment/train[0m
    💡 Learn more at https://docs.ultralytics.com/modes/train


### Using / Testing the Model


```python
!yolo task=segment mode=predict model=/content/runs/segment/train/weights/best.pt conf=0.85 source={dataset.location}/*.jpg
```

    Ultralytics YOLOv8.0.222 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    YOLOv8x-seg summary (fused): 295 layers, 71723545 parameters, 0 gradients, 343.7 GFLOPs
    
    image 1/15 /content/vehicle-segmentation-1/car1.jpg: 640x640 1 car, 73.9ms
    image 2/15 /content/vehicle-segmentation-1/car2.jpg: 384x640 1 car, 71.1ms
    image 3/15 /content/vehicle-segmentation-1/car3.jpg: 416x640 1 car, 90.5ms
    image 4/15 /content/vehicle-segmentation-1/car4.jpg: 640x640 1 car, 75.2ms
    image 5/15 /content/vehicle-segmentation-1/car5.jpg: 640x640 1 car, 75.0ms
    image 6/15 /content/vehicle-segmentation-1/pickup1.jpg: 640x640 1 pickup, 73.2ms
    image 7/15 /content/vehicle-segmentation-1/pickup2.jpg: 576x640 1 pickup, 73.5ms
    image 8/15 /content/vehicle-segmentation-1/pickup3.jpg: 640x640 1 pickup, 64.3ms
    image 9/15 /content/vehicle-segmentation-1/pickup4.jpg: 384x640 1 pickup, 38.4ms
    image 10/15 /content/vehicle-segmentation-1/pickup5.jpg: 480x640 1 pickup, 77.5ms
    image 11/15 /content/vehicle-segmentation-1/truck1.jpg: 640x640 1 truck, 65.6ms
    image 12/15 /content/vehicle-segmentation-1/truck2.jpg: 480x640 1 truck, 45.6ms
    image 13/15 /content/vehicle-segmentation-1/truck3.jpg: 640x640 1 truck, 63.6ms
    image 14/15 /content/vehicle-segmentation-1/truck4.jpg: 448x640 1 truck, 79.0ms
    image 15/15 /content/vehicle-segmentation-1/truck5.jpg: 640x640 1 truck, 66.7ms
    Speed: 2.9ms preprocess, 68.9ms inference, 2.2ms postprocess per image at shape (1, 3, 640, 640)
    Results saved to [1mruns/segment/predict[0m
    💡 Learn more at https://docs.ultralytics.com/modes/predict



```python
import glob
from IPython.display import Image, display

for image_path in glob.glob(f'/content/runs/segment/predict/*.jpg'):
  display(Image(filename=image_path, height=600))
  print("\n")
```


    
![output_11_0](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_0.jpg)
    
![output_11_1](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_1.jpg)

![output_11_2](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_2.jpg)

![output_11_3](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_3.jpg)

![output_11_4](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_4.jpg)

![output_11_6](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_6.jpg)

![output_11_8](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_8.jpg)

![output_11_10](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_10.jpg)

![output_11_12](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_12.jpg)

![output_11_14](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_14.jpg)

![output_11_16](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_16.jpg)

![output_11_18](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_18.jpg)

![output_11_20](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_20.jpg)

![output_11_22](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_22.jpg)

![output_11_24](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/output/output_11_24.jpg)

### Evaluation


```python
# Confusion matrix

Image(filename=f"/content/runs/segment/train/confusion_matrix.png", width=600)
```




    
![confusion_matrix](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/confusion_matrix.png)
    




```python
Image(filename="/content/runs/segment/train/results.png", width=600)
```




    
![results](https://github.com/kemalkilicaslan/Vehicle_Recognition_with_Segmentation_Training_on_a_Custom_Dataset/blob/main/results.png)
    




```python
!yolo task=segment mode=val model=/content/runs/segment/train/weights/best.pt data={dataset.location}/data.yaml
```

    Ultralytics YOLOv8.0.222 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    YOLOv8x-seg summary (fused): 295 layers, 71723545 parameters, 0 gradients, 343.7 GFLOPs
    [34m[1mval: [0mScanning /content/vehicle-segmentation-1/valid/labels.cache... 12 images, 0 backgrounds, 0 corrupt: 100% 12/12 [00:00<?, ?it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% 1/1 [00:01<00:00,  1.14s/it]
                       all         12         12      0.916      0.862      0.978      0.869      0.916      0.862      0.978      0.816
                       car         12          4      0.783          1      0.995      0.921      0.783          1      0.995      0.958
                    pickup         12          4      0.964          1      0.995      0.786      0.964          1      0.995      0.663
                     truck         12          4          1      0.586      0.945      0.899          1      0.586      0.945      0.828
    Speed: 0.4ms preprocess, 83.7ms inference, 0.0ms loss, 0.9ms postprocess per image
    Results saved to [1mruns/segment/val[0m
    💡 Learn more at https://docs.ultralytics.com/modes/val



```python
import pandas as pd

precision_values = {'all': 0.916, 'car': 0.783, 'pickup': 0.964, 'truck': 1.000}
recall_values = {'all': 0.862, 'car': 1.000, 'pickup': 1.000, 'truck': 0.586}

# F1 Score calculation function
def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

data = []

# Calculate F1 Score for each class and add it to the table
for class_name in precision_values.keys():
    precision = precision_values[class_name]
    recall = recall_values[class_name]
    f1_score = calculate_f1_score(precision, recall)

    data.append({
        'Class': class_name,
        'Box(Precision)': precision,
        'Box(Recall)': recall,
        'F1 Score': f1_score
    })

df = pd.DataFrame(data)
print(df)
```

        Class  Box(Precision)  Box(Recall)  F1 Score
    0     all           0.916        0.862  0.888180
    1     car           0.783        1.000  0.878295
    2  pickup           0.964        1.000  0.981670
    3   truck           1.000        0.586  0.738966