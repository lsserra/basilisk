import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
PATH2SIMDATADIR = "/Users/lukeserrano/repos/personal/basilisk/prjs/aero626/data"


runNum = 55
pth2data= os.path.join(PATH2SIMDATADIR,f"sim-00{runNum}")




from prjs.aero626.analysis.AnalysisTools import PoseAnalyzer

egmfPA = PoseAnalyzer()
pklPath = os.path.join(pth2data,"EGMF.pkl")
egmfPA._solutionName = "EGMF"
egmfPA.LoadFromPklFile(pkl_path=pklPath)
egmfPA.plotTransError3sigma()
egmfPA.plotAttError3Sigma()

ekfPA = PoseAnalyzer()
pklPath = os.path.join(pth2data,"EKF.pkl")
ekfPA._solutionName = "EKF"
# ekfPA._NO_TITLE_FLAG = True
ekfPA.LoadFromPklFile(pkl_path=pklPath)
ekfPA.plotTransError3sigma()
ekfPA.plotAttError3Sigma()




plt.show()

