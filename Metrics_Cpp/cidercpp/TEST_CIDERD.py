"""
from cider.pyciderevalcap.cider.cider import Cider as C_py

res = [{'caption':['i have just brought a banana'],'image_id':'1'}, {'caption':['safe mode limited country'], 'image_id':'2'}]
gts1 = {'1': ['i have just brought a yellow banana'], '2':['safe mode limited country']}
gts2 = {'1': ['i have just brought a yellow banana', 'a yellow banana looks blue'], '2':['safe mode limited country']}

c_Py = C_py()
print(c_Py.compute_score(gts1, res))
print(c_Py.compute_score(gts2, res))


from cidercpp.CiderCpp import Cider as C_cpp
resC = ['i have just brought a banana','safe mode limited country']
gts1C = [['i have just brought a yellow banana'],['safe mode limited country']]
gts2C = [['i have just brought a yellow banana', 'a yellow banana looks blue'],['safe mode limited country']]
c_Cpp = C_cpp(4,6)
print(c_Cpp.compute_score(gts1C, resC))
print(c_Cpp.compute_score(gts2C, resC))
"""


from cider.pyciderevalcap.ciderD.ciderD import CiderD as C_py

res = [{'caption':['i have just brought a banana'],'image_id':'1'}, {'caption':['safe mode limited country'], 'image_id':'2'}]
gts1 = {'1': ['i have just brought a yellow banana'], '2':['safe mode limited country']}
gts2 = {'1': ['i have just brought a yellow banana', 'a yellow banana looks blue'], '2':['safe mode limited country']}

c_Py = C_py([['i have just brought a yellow banana','a yellow banana looks blue'],['safe mode limited country']], 4, 6)
print(c_Py.compute_score(gts1, res))
print(c_Py.compute_score(gts2, res))


from ciderDcpp.CiderDCpp import Cider as C_cpp
resC = ['i have just brought a banana','safe mode limited country']
gts1C = [['i have just brought a yellow banana'],['safe mode limited country']]
gts2C = [['i have just brought a yellow banana', 'a yellow banana looks blue'],['safe mode limited country']]
c_Cpp = C_cpp([['i have just brought a yellow banana','a yellow banana looks blue'],['safe mode limited country']], 4, 6)
print(c_Cpp.compute_score(gts1C, resC))
print(c_Cpp.compute_score(gts2C, resC))
