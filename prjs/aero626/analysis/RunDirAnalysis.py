import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
PATH2SIMDATADIR = "/Users/lukeserrano/repos/personal/basilisk/prjs/aero626/data"
pth2data= os.path.join(PATH2SIMDATADIR,"sim-0012/FILTER_SOL_AND_SIM_TRUTH.pkl")




from prjs.aero626.analysis.AnalysisTools import PoseAnalyzer

pa = PoseAnalyzer()
pa.LoadFromPklFile(pkl_path=pth2data)
pa.plotTransError3sigma()
plt.show()