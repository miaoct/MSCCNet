# MSCCNet
Multi-spectral Class Center Network for Face Manipulation Detection and Localization

***
### Annotation
First, you can utilize MTCNN to detect and crop real face images. <br>

Then, you can apply the bounding box of the real face to crop the corresponding fake face image.  <br>
This ensures that both the real and corresponding fake face images have the same size.  <br>
Our real-fake pairs list in the *./annotation/real_fake_list/* .  <br>

Lastly, run ` python annotation_ff.py`.  <br>

***
### Train
Run `bash distrain_valid.sh`

***
### Test
Run `bash distest.sh`

***
### Acknowledge
The code framework is based on [SSSegmentation](https://github.com/SegmentationBLWX/sssegmentation), thanks for Zhenchao Jin's help with the code. <br>
Please see the document of SSSegmentation to **Install Requirements**.
