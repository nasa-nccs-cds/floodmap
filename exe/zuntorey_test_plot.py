import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "/Users/tpmaxwel/.floodmap/test/data/zun-torey.txt"
data_file = open(data_path)

figure, ax = plt.subplots()

time, value = [], []
for line in data_file.readlines():
    toks = line.split(' ')
    if len( toks ) > 1:
        time.append( pd.to_datetime(toks[0]) )
        value.append( float( toks[1] ) )

times = np.array( time, dtype=np.datetime64 )
values = np.array( value )
ax.plot( times, values, "r-" )
plt.title('Zun Torey')
ax.set_ylabel('Water Surface Area (km2)')
ax.set_xlabel('Date')
plt.ylim([0,160])
plt.show()
