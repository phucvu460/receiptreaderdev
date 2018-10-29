# TextBoxes: A Fast Text Detector with a Single Deep Neural Network

Recommend: [TextBoxes++](https://github.com/MhLiao/TextBoxes_plusplus) is an extended work of TextBoxes, which supports oriented scene text detection. The recognition part is also included in [TextBoxes++](https://github.com/MhLiao/TextBoxes_plusplus).

### Introduction
This paper presents an end-to-end trainable fast scene text detector, named TextBoxes, which detects scene text with both high accuracy and efficiency in a single network forward pass, involving no post-process except for a standard nonmaximum suppression. For more details, please refer to our [paper](https://arxiv.org/abs/1611.06779).

### Citing TextBoxes
Please cite TextBoxes in your publications if it helps your research:

    @inproceedings{LiaoSBWL17,
      author    = {Minghui Liao and
                   Baoguang Shi and
                   Xiang Bai and
                   Xinggang Wang and
                   Wenyu Liu},
      title     = {TextBoxes: {A} Fast Text Detector with a Single Deep Neural Network},
      booktitle = {AAAI},
      year      = {2017}
    }


### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Test](#test)
4. [Train](#train)
5. [Performance](#performance)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/MhLiao/TextBoxes.git
  
  cd TextBoxes
  
  make -j8
  
  make py
  ```

### Download
1. Models trained on ICDAR 2013: [Dropbox link](https://www.dropbox.com/s/g8pjzv2de9gty8g/TextBoxes_icdar13.caffemodel?dl=0) [BaiduYun link](http://pan.baidu.com/s/1qY73XHq)
2. Fully convolutional reduced (atrous) VGGNet: [Dropbox link](https://www.dropbox.com/s/qxc64az0a21vodt/VGG_ILSVRC_16_layers_fc_reduced.caffemodel?dl=0) [BaiduYun link](http://pan.baidu.com/s/1slQyMiL)
3. Compiled mex file for evaluation(for multi-scale test evaluation: evaluation_nms.m): [Dropbox link](https://www.dropbox.com/s/xtjuwvphxnz1nl8/polygon_intersect.mexa64?dl=0) [BaiduYun link](http://pan.baidu.com/s/1jIe9UWA)


### Test
1. run "python examples/demo.py".
2. You can modify the "use_multi_scale" in the "examples/demo.py" script to control whether to use multi-scale or not.
3. The results are saved in the "examples/results/".


### Train
1. Train about 50k iterions on Synthetic data which refered in the paper.
2. Train about 2k iterions on corresponding training data such as ICDAR 2013 and SVT.
3. For more information, such as learning rate setting, please refer to the paper.

### Performance
1. Using the given test code, you can achieve an F-measure of about 80% on ICDAR 2013 with a single scale.
2. Using the given multi-scale test code, you can achieve an F-measure of about 85% on ICDAR 2013 with a non-maximum suppression.
3. More performance information, please refer to the paper and Task1 and Task4 of Challenge2 on the ICDAR 2015 website: http://rrc.cvc.uab.es/?ch=2&com=evaluation

### Data preparation for training
The reference xml file is as following:
  
        <?xml version="1.0" encoding="utf-8"?>
        <annotation>
            <object>
                <name>text</name>
                <bndbox>
                    <xmin>158</xmin>
                    <ymin>128</ymin>
                    <xmax>411</xmax>
                    <ymax>181</ymax>
                </bndbox>
            </object>
            <object>
                <name>text</name>
                <bndbox>
                    <xmin>443</xmin>
                    <ymin>128</ymin>
                    <xmax>501</xmax>
                    <ymax>169</ymax>
                </bndbox>
            </object>
            <folder></folder>
            <filename>100.jpg</filename>
            <size>
                <width>640</width>
                <height>480</height>
                <depth>3</depth>
            </size>
        </annotation>

Please let me know if you encounter any issues.
