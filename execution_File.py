from Uncertain_EP import *


# Execution
#from FstRun.Uncertain_EP import Uncertain_EP

testing = Uncertain_EP('C:\\Users\\YunJoon Jung\Dropbox (GaTech)\\Uncertain_EP\Mohanned\\MyHouse_Monthlt_Temp_afterCal.idf', 'C:\\Users\\YunJoon Jung\Dropbox (GaTech)\\Uncertain_EP\\Mohanned\\Atlanta2018.epw',
                       'C:\\Users\\YunJoon Jung\Dropbox (GaTech)\\Uncertain_EP\Mohanned\\Energy+.idd', climate_uncertainty=True, SA_Graph=True, UA_Graph=True)

##testing.SA(number_of_SA_samples=3)
testing.UA(number_of_UA_samples=3, only_idf_instances_generation=False)
