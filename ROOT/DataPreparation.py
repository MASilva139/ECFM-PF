import ROOT

def filter_events(df):
    """
    Reduce initial dataset to only events which shall be used for training
    """
    return df.Filter("Jet_size>2 && ( Electron_size==0 && Muon_size>=1 )", "At least four jets and one electron or muon")
 
 
def define_jet_variables(df):
    """
    Define the variables which shall be used for training
    """
    
    return df.Define("Muon_px","Muon.PT[0]*cos(Muon.Phi[0])")\
             .Define("Muon_py","Muon.PT[0]*sin(Muon.Phi[0])")\
             .Define("Muon_pz","Muon.PT[0]*sinh(Muon.Eta[0])")\
             .Define("Muon_E","Muon.PT[0]*cosh(Muon.Eta[0])")

variables = ["Muon_px","Muon_py","Muon_pz","Muon_E"]


#######################################################################

if __name__ == "__main__":
    for filename, label in [["meissa_ggf-h-ww-jjlnu_*_delphes_events_withCuts.root", "signal"], ["meissa_pp-jjlv_*_delphes_events_withCuts.root", "background"], ["meissa_pp-jjlnu_*_delphes_events_withCuts.root", "background"]]:

        print(">>> Extract the training and testing events for {} from the {} dataset.".format(
            label, filename))
 
        # Load dataset, filter the required events and define the training variables
        filepath = "../data/" + filename
        df = ROOT.RDataFrame("Delphes", filepath)
        df = filter_events(df)
        df = define_jet_variables(df)
 
        # Book cutflow report
        report = df.Report()
 
        # Split dataset by event number for training and testing
        columns = ROOT.std.vector["string"](variables)
#        df.Filter("Event.Number % 2 == 0", "Select events with even event number for training")\
        df.Range(0,0,2)\
          .Snapshot("Events", "./train_" + label + ".root", columns)
#        df.Filter("Event_size % 2 == 1", "Select events with odd event number for test")\
        df.Range(1,0,2)\
          .Snapshot("Events", "./test_" + label + ".root", columns)
 
        # Print cutflow report
        report.Print()
