# Metrics written in Cpp

Calculating Cider and BLEU with python is slow. I rewrite these python codes into Cpp, which makes the training 2x faster.

# Usage

First you need to compile Cpp files, then import this module in python.

**To compile these Cpp codes**

Compile Cpp for CiderD for example:

```
swig -c++ -python CiderD.i
python setup.py build_ext --inplace
```

**To test these codes**

```
from CiderDCpp import Cider as C_cpp
resC = ['i have just brought a banana','safe mode limited country']
gts1C = [['i have just brought a yellow banana'],['safe mode limited country']]
gts2C = [['i have just brought a yellow banana', 'a yellow banana looks blue'],['safe mode limited country']]
c_Cpp = C_cpp([['i have just brought a yellow banana','a yellow banana looks blue'],['safe mode limited country']], 4, 6)
print(c_Cpp.compute_score(gts1C, resC))
print(c_Cpp.compute_score(gts2C, resC))
```

Output:

```
(8.580286026000977, (7.160572528839111, 10.0))
(7.0152130126953125, (4.030426025390625, 10.0))
```