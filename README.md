#####LibN3L: A light-weight neural network package for natural language.

Just include the directory in your code and call it by "#include N3L.h" 

Prerequisition:  
&ensp;&ensp;&ensp;&ensp;***mshadow***  
Please download and include the directory  "https://github.com/dmlc/mshadow/tree/master/mshadow" in your applications:  
(a) first copy the directory into your computer;  
(b) then include it in your applications;  
If have any problems, please mason.zms@gmail.com  

If you want to make mshadow work, you need to install certain libaries such as openblas and cuda.  
I suggest use **openblas** since the current version does not support cuda yet, which is our future work.  
Find it here:  
   https://github.com/xianyi/OpenBLAS  
Compile and install:  
make USE_THREAD=0 &ensp;&ensp;&ensp;&ensp;##single thread version, one can use multi-thread version as well.  
make   install&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;##default path /opt/OpenBLAS  
cp  /opt/OpenBLAS/include/*.*  /usr/include/  
cp  /opt/OpenBLAS/lib/*.*     /usr/lib(64)/  


Some examples are realeased at:  
https://github.com/SUTDNLP/NNSegmentation  
https://github.com/SUTDNLP/NNPOSTagging  
https://github.com/SUTDNLP/NNNamedEntity  
You can see the performances in **description.pdf**  
https://github.com/SUTDNLP/OpenTargetedSentiment  
The code of my EMNLP2015 paper:  
Meishan Zhang; Yue Zhang; Duy Tin Vo. [Neural Networks for Open Domain Targeted Sentiment.](http://www.aclweb.org/anthology/D/D15/D15-1073.pdf) EMNLP2015.  
