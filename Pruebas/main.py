"""B+/- -> J/psi K+/- analysis example for LHCb collision ntuples.

This analysis example demonstrates how to work with LHCb collision
ntuples produced by the LHCb Ntupling Service. The example studies
B+/- -> J/psi K+/- decay and demonstrates plotting of the B+/- meson
candidate mass and J/psi candidate mass.
"""

import ROOT
ROOT.gROOT.SetBatch(True)

root_files = [
    "data/00334560_00000001_1.dvntuple.root",
    "data/00334560_00000002_1.dvntuple.root",
    "data/00334564_00000001_1.dvntuple.root",
    "data/00334565_00000001_1.dvntuple.root",
    "data/00334566_00000001_1.dvntuple.root",
]
rdf = ROOT.RDataFrame("Btree/DecayTree", root_files)

hist_args_B = (
    "",
    "B candidate mass;#it{m}_{#mu^{+}#mu^{-}K^{#pm}};candidates",
    200,
    4600,
    6000,
)
hist_B = rdf.Histo1D(hist_args_B, "Bplus_M")

hist_args_jpsi = (
    "",
    "J/#psi candidate mass;#it{m}_{#mu^{+}#mu^{-} [MeV/c^{2}]};candidates",
    200,
    2950,
    3250,
)
hist_jpsi = rdf.Histo1D(hist_args_jpsi, "J_psi_1S_M")

canvas = ROOT.TCanvas("canvas", "", 900, 600)
hist_B.Draw()
canvas.SaveAs("Pruebas/figs/bmass.png")
hist_jpsi.Draw()
canvas.SaveAs("Pruebas/figs/jpsimass.png")