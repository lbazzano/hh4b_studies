n_bins = 100
maximum = 400
dens = True

h,b,p = plt.hist(df.pt0,bins=np.linspace(0,maximum,n_bins),weights=df.weight,density=dens,align='mid',label="weighted QCD",alpha=0.7,color="tab:red")
bin_w = (b[1]-b[0])
centers = b[1:]-bin_w/2
#plt.plot(centers,h)
fit = np.poly1d(np.polyfit(centers, h, 100))
plt.plot(centers,fit(centers),label="fit",color="red")
plt.yscale('log')
h,b,p = plt.hist(df.pt0,bins=np.linspace(0,maximum,n_bins),weights=df.weight*(1/fit(df.pt0)),density=dens,align='mid',label="flat QCD",alpha=0.3,color="tab:green")
signal_ind = df.s_or_b > 0.5
h,b,p = plt.hist(df.pt0[signal_ind],bins=np.linspace(0,maximum,n_bins),weights=df.weight[signal_ind]*(1/fit(df.pt0[signal_ind])),density=dens,align='mid',label="flat signal QCD",histtype="step",linewidth=2,color="orange")
back_ind = df.s_or_b < 0.5
h,b,p = plt.hist(df.pt0[back_ind],bins=np.linspace(0,maximum,n_bins),weights=df.weight[back_ind]*(1/fit(df.pt0[back_ind])),density=dens,align='mid',label="flat background QCD",histtype="step",linewidth=2,color="blue")
plt.legend()
plt.grid()
plt.show()

