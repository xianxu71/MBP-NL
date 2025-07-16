import h5py as h5
import numpy as np
from lxml import etree
from constants import Ry2eV, a2bohr

"""
Build wfn.h5 header from a vasp calculations
parse vasprun.xml from https://github.com/qzhu2017/vasprun/

"""
def saveh5(values, fname='wfn.h5'):
   """
   """
   kpoints = np.array(values["kpoints"]["list"])
   w       = np.array(values["kpoints"]["weights"])
   kgrid   = np.array(values["kpoints"]["divisions"], dtype=np.int32)
   tmp     = np.array(values["calculation"]["eband_eigenvalues"])
   nk      = kpoints.shape[0]

   nspin = values["nspin"]
   nband = values["parameters"]["electronic"]["NBANDS"]
   tmp = tmp.reshape([nspin,nk,nband,2]) 
   el = tmp[...,0]/Ry2eV
   occ = tmp[...,1]

   avec = np.array(values["finalpos"]["basis"])*a2bohr
   bvec = np.array(values["finalpos"]["rec_basis"])/a2bohr
   bdot = np.dot(bvec,bvec.T)

   ifmax = np.zeros([nspin,nk],dtype=np.int32)
   for js in range(nspin):
     for ik in range(nk):
       ifmax[js,ik] = np.sum(occ[js,ik])

   with h5.File(fname, 'w') as f:
     kpg = f.create_group("mf_header/kpoints")

     kpg.create_dataset("nspin", data=nspin)
     kpg.create_dataset("nspinor", data=values["nspinor"])
     kpg.create_dataset("occ", data=occ)
     kpg.create_dataset("ifmax", data=ifmax)
     kpg.create_dataset("el", data=el)
     kpg.create_dataset("mnband", data=nband)

     kpg.create_dataset('rk', data=kpoints)     
     kpg.create_dataset('nrk', data=nk)        
     kpg.create_dataset('kgrid', data=kgrid)
     kpg.create_dataset('w', data=w)             
                            
     # FIX: symmetry data is not read                             
     f.create_dataset('mf_header/symmetry/mtrx',\
                         data=np.zeros([48,3,3],dtype=np.int32))    
     f.create_dataset('mf_header/symmetry/ntran', data=1)    
     f.create_dataset('mf_header/symmetry/tnp',\
                         data=np.zeros([48,3],dtype=float))

     crg = f.create_group("mf_header/crystal")

     crg.create_dataset("celvol", data=values["volume"]*a2bohr**3)
     crg.create_dataset("alat", data=1.0)
     crg.create_dataset("avec", data=avec)
     crg.create_dataset('bdot', data=bdot) 
     crg.create_dataset('bvec', data=bvec) 
     crg.create_dataset('blat', data=1.0) 

   return

def parse_varray_pymatgen(elem):
   """
   """
   def _vasprun_float(f):
       """
       Large numbers are often represented as ********* in the vasprun.
       This function parses these values as np.nan
       """
       try:
           return float(f)
       except ValueError as e:
           f = f.strip()
           if f == '*' * len(f):
               warnings.warn('Float overflow (*******) encountered in vasprun')
               return np.nan
           raise e
   if elem.get("type", None) == 'logical':
       m = [[True if i == 'T' else False for i in v.text.split()] for v in elem]
   else:
       m = [[_vasprun_float(i) for i in v.text.split()] for v in elem]

   return m

def parse_varray(varray):
   """
   """
   if varray.get("type") == 'int':
       m = [[int(number) for number in v.text.split()] for v in varray.findall("v")]
   else:
       try:
           m = [[float(number) for number in v.text.split()] for v in varray.findall("v")]
       except:
           m = [[0 for number in v.text.split()] for v in varray.findall("v")]
   return m

def parse_eigenvalue(eigenvalue):
   """
   """
   eigenvalues = []
   for s in eigenvalue.find("array").find("set").findall("set"):
       for ss in s.findall("set"):
           eigenvalues.append(parse_varray_pymatgen(ss))
   return eigenvalues

def parse_kpoints(kpoints):
   """
   """
   kpoints_dict = {'list': [], 'weights': [], 'divisions': [], 'mesh_scheme': ''}

   for i in kpoints.iterchildren():
       if i.tag == 'generation':
           kpoints_dict['mesh_scheme'] = i.attrib.get('param')
           for j in i.iterchildren():
               if j.attrib.get("name") == 'divisions':
                   kpoints_dict['divisions'] = [int(number) for number in j.text.split()]
                   break

   for va in kpoints.findall("varray"):
       name = va.attrib["name"]
       if name == "kpointlist":
           kpoints_dict['list'] = parse_varray(va)
       elif name == "weights":
           kpoints_dict['weights'] = parse_varray(va)

   return kpoints_dict

def parse_parameters(children):
   """
   """
   parameters = {}
   for i in children:
     if i.tag == "separator":
         name = i.attrib.get("name")
         d = parse_i_tag_collection(i)
         parameters[name] = d
         for ii in i:
             if ii.tag == "separator":
                 name2 = ii.attrib.get("name")
                 d2 = parse_i_tag_collection(ii)
                 parameters[name][name2] = d2

   return parameters

def parse_volume(children):
   """
   """
   for i in children:
      if i.tag == "crystal":
       for j in i.findall("i"):
           if j.attrib.get("name") == "volume":
               volume = float(j.text)
               break
   
   return volume

def parse_basis(children):
   """
   """
   d = {}
   for i in children.iter("varray"):
       name = i.attrib.get("name")
       d[name] = parse_varray(i)

   return d

def parse_i_tag_collection(itags_collection):
   """
   """
   d = {}
   for info in itags_collection.findall("i"):
       name = info.attrib.get("name")
       type = info.attrib.get("type")
       content = info.text
       d[name] = assign_type(type, content)
   return d

def assign_type(type, content):
   """
   """
   if type == "logical":
       content = content.replace(" ", "")
       if content in ('T', 'True', 'true'):
           return True
       elif content in ('F', 'False', 'false'):
           return False
       else:
           Warning("logical text " + content + " not T, True, true, F, False, false, set to False")
       return False
   elif type == "int":
       return int(content) if len(content.split()) == 1 else [int(number) for number in content.split()]
   elif type == "string":
       return content
   elif type is None:
       return float(content) if len(content.split()) == 1 else [float(number) for number in content.split()]
   else:
       Warning("New type: " + type + ", set to string")
   return content

def parse_calculation(calculation):

   for i in calculation.iterchildren():
       if i.attrib.get("name") == "stress":
           stress = parse_varray(i)
       elif i.attrib.get("name") == "forces":
           force = parse_varray(i)
       elif i.tag == "dos":
           for j in i.findall("i"):
               if j.attrib.get("name") == "efermi":
                   efermi = float(j.text)
                   break
       elif i.tag == "eigenvalues":
           eigenvalues = parse_eigenvalue(i)
       elif i.tag == "energy":
           for e in i.findall("i"):
               if e.attrib.get("name") == "e_fr_energy":
                   try:
                       energy = float(e.text)
                   except ValueError:
                       energy = 1000000000
               else:
                   Warning("No e_fr_energy found in <calculation><energy> tag, energy set to 0.0")

   calculation = {} 
   calculation["efermi"] = efermi
   calculation["eband_eigenvalues"] = eigenvalues

   return calculation

def parse_vasp(fname='vasprun.xml'):
   """

   """
   values = {}

   doc = etree.parse(fname)
   root = doc.getroot()

   for child in root.iterchildren():
      if child.tag == "kpoints":
        values["kpoints"] = parse_kpoints(child)
      elif child.tag == "parameters":
        values[child.tag] = parse_parameters(child)
      elif child.tag == "calculation":
        values["calculation"] = parse_calculation(child)
      elif child.tag == "structure" and child.attrib.get("name") == "finalpos":
        values["volume"] = parse_volume(child)
        values["finalpos"] = parse_basis(child)

   if values['parameters']['electronic']['electronic spin']['LSORBIT']:
      values["nspinor"] = 2
   else:
      values["nspinor"] = 1

   if values['parameters']['electronic']['electronic spin']['ISPIN'] == 2:
      values["nspin"] = 2
   else:
      values["nspin"] = 1

   return values

if __name__ == '__main__':

   values = parse_vasp()
   saveh5(values)
