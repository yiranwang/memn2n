import h5py

f = h5py.File('model.h5', mode='r')
print f.attrs['layer_names']
