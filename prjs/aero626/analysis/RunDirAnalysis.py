import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
PATH2SIMDATADIR = "/Users/lukeserrano/repos/personal/basilisk/prjs/aero626/data"


runNum = 14
pth2data= os.path.join(PATH2SIMDATADIR,f"sim-00{runNum}")




from prjs.aero626.analysis.AnalysisTools import PoseAnalyzer

egmfPA = PoseAnalyzer()
pklPath = os.path.join(pth2data,"EGMF.pkl")
egmfPA.LoadFromPklFile(pkl_path=pklPath)
egmfPA.plotTransError3sigma()

ekfPA = PoseAnalyzer()
pklPath = os.path.join(pth2data,"EKF.pkl")
ekfPA.LoadFromPklFile(pkl_path=pklPath)
ekfPA.plotTransError3sigma()



plt.show()

