1. manually selected good frontal faces from att dataset and organized to 4 folders in root directory (e.g.F:\face\my_collection): male, male with glass, female, female with glass.

2. call blend_att.py to generate class 1 samples (tampered faces); class 0 samples may include some head pose/expression variants

3. call blend_yale.py to generate class 1 samples from *P00A+000E+00.pgm images; class 0 samples include lighting changes

4. call preprocess_data.py to remove black boundaries from blending.

Note: need dlib and [imutils](https://github.com/jrosebr1/imutils)
