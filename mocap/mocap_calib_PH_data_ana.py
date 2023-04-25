import pickle

raw_data_dir = 'PH_raw_data/train_data'
with open(raw_data_dir+'_'+str(1)+'.pickle', 'rb') as handle:
    curve_p = pickle.load(handle)

print(curve_p['marker1_rigid3'][0])