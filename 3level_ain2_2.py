from Frequency_response_3level import *
delovals=np.linspace(-20e5,20e5,9)
delmvals=np.linspace(-1e6,1e6,81)
binvals=[0]
ainvals=[600000]
aoutvals,boutvals,effic_a,effic_b=find_outputs(ainvals,binvals,delovals,delmvals)
np.savez('output_ain2_2',aoutvals=aoutvals,boutvals=boutvals,effic_a=effic_a,effic_b=effic_b,ainvals=ainvals,binvals=binvals,delovals=delovals,delmvals=delmvals)