from Frequency_response_3level import *
delovals=np.linspace(-20e6,20e6,101)
delmvals=np.linspace(-2e6,2e6,11)
binvals=[0]
ainvals=[600000]
aoutvals,boutvals,effic_a,effic_b=find_outputs(ainvals,binvals,delovals,delmvals)
np.savez('output_ain1',aoutvals=aoutvals,boutvals=boutvals,effic_a=effic_a,effic_b=effic_b,ainvals=ainvals,binvals=binvals,delovals=delovals,delmvals=delmvals)
