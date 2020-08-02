# Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence[Lee+, CVPR20]
https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Reference-Based_Sketch_Image_Colorization_Using_Augmented-Self_Reference_and_Dense_Semantic_CVPR_2020_paper.pdf

**Note that this is an ongoing re-implementation and I cannot fully reproduce the results. Suggestions and PRs are welcome!**

# Requirements
+ Python 3.6+
+ PyTorch 0.4+

# Usage
1. Download [a Tag2Pix dataset from the officical repsitory.](https://github.com/blandocs/Tag2Pix).
2. Put it on `./datasets/tag2pix`
3. Run `bash scripts/train_tag2pix_xdog.sh baseline`. The training using sketches by XDoG will run.
4. Run `bash scripts/train_tag2pix_keras.sh baseline`. The training using sketches by SketchKeras will run.


# LICENCE
All code is licensed under the MIT license.

# Acknowledgements
This repository is based on https://github.com/yunjey/stargan.

Additionally, if you use this repository, please cite original paper
```
@InProceedings{lee2020referencebased,
    title={Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence},
    author={Junsoo Lee and Eungyeup Kim and Yunsung Lee and Dongjun Kim and Jaehyuk Chang and Jaegul Choo},
    year={2020},
    booktitle = {Proc. IEEE Computer Vision and Pattern Recognition (CVPR)}
}
```