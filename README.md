# LibN3L
A light-weight neural network package for natural language.

Just include the directory in your code and call it by "#include N3L.h" 

Prerequisition:
    mshadow

Please include the directory of "https://github.com/dmlc/mshadow/tree/master/mshadow" in your applications.
(First copy the directory into your computer, then include it in your applications)

If you want to make mshadow work, you need install certain libaries for example openblas.
I suggest use openblas since the current version does not support cuda yet, which is our future work.
Find it here:
   https://github.com/xianyi/OpenBLAS
make USE_THREAD=0                             ###(single thread version) 
make install                                  ###(default path /opt/)
cp  /opt/OpenBLAS/include/*.*  /usr/include/
cp  /opt/OpenBLAS/lib/*.*     /usr/lib(64)/  



Some examples are realeased at:
https://github.com/SUTDNLP/NNSegmentation
https://github.com/SUTDNLP/NNPOSTagging
https://github.com/SUTDNLP/NNNamedEntity
You can see the performances in description.pdf
https://github.com/SUTDNLP/OpenTargetedSentiment