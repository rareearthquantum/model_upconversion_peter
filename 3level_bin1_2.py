from Frequency_response_3level import *
delovals=np.linspace(-20e5,20e5,81)
delmvals=np.linspace(-1e6,1e6,9)
ainvals=[0]
binvals=[600000]
aoutvals,boutvals,effic_a,effic_b=find_outputs(ainvals,binvals,delovals,delmvals)
np.savez('output_bin1_2',aoutvals=aoutvals,boutvals=boutvals,effic_a=effic_a,effic_b=effic_b,ainvals=ainvals,binvals=binvals,delovals=delovals,delmvals=delmvals)
