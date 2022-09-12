import os 
import numpy as np
gammas = np.linspace(110., 10000, 32)
for g in gammas:
    path_tar="/nfs/pic.es/user/m/mbilkis/many_fig/{}".format(g)
    os.makedirs(path_tar,exist_ok=True)
    os.system("cp -r {}/T_8.0_dt_0.0001/figures/* ".format(g) + path_tar)
    print(g)
os.system("zip -r /nfs/pic.es/user/m/mbilkis/many_fig.zip /nfs/pic.es/user/mbilkis/many_fig")
#os.system("rm -r /nfs/pic.es/user/m/mbilkis/many_fig")
