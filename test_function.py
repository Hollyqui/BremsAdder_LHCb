# from itertools import compress
#
# def filter(x, bool_list):
#     return list(compress(x, bool_list))
#
#
# a = [1,2,3,4,5,6,7,8,9]
# b = [True, False, False, False, True, True, False, False, True, True]
# b = [not i for i in b]
# c = filter(a, b)
# print(c)


import ROOT
from array import array

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(55)

f = ROOT.TFile.Open("/home/felix/PycharmProjects/Thesis/Data/data/DaVinci_PgunElectrons_081119_1000ev.root")
tree = f.Get("electrons/DecayTree")

cECAL = ROOT.TCanvas("cECAL")

for entry in tree:

    clusterxlist = entry.ECAL_cluster_x_arr
    clusterylist = entry.ECAL_cluster_y_arr
    clusterelist = entry.ECAL_cluster_e_arr

    hECAL = ROOT.TH2F("hECAL", "ECAL clusters", 30, -3000, 3000, 30, -3000, 3000)
    for i in range(entry.N_ECAL_clusters):
        # print("x={0}, y={1}, e={2}".format(clusterxlist[i], clusterylist[i], clusterelist[i] ))
        hECAL.Fill(clusterxlist[i], clusterylist[i], clusterelist[i])

    hECAL.GetXaxis().SetTitle("x [mm]")
    hECAL.GetYaxis().SetTitle("y [mm]")
    hECAL.GetZaxis().SetTitle("E")
    hECAL.SetMinimum(0.)
    hECAL.SetMaximum(50000.)
    hECAL.Draw("COLZ")
    cECAL.SetFixedAspectRatio()

    electron_x = entry.eminus_ECAL_x
    electron_y = entry.eminus_ECAL_y
    electrongraph = ROOT.TGraph(1, array('f', [electron_x]), array('f', [electron_y]))
    electrongraph.SetMarkerStyle(43)
    electrongraph.SetMarkerSize(2)
    electrongraph.Draw("Psame")

    cECAL.Update()

    hasbrem = entry.eminus_HasBremAdded
    input(
        "Electron is at x = {0:.0f}, y = {1:.0f} - hasBremAdded = {2}. (Press enter for next event)".format(electron_x,
                                                                                                            electron_y,
                                                                                                            hasbrem))

    hECAL.Delete()

f.Close()