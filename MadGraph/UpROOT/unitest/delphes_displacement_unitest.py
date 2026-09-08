import unittest
import numpy as np
import pandas as pd
from UpROOT.physics import delphes

class DelphesDisplacementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dimuons = pd.DataFrame({
            "event_id": [1, 2],
            "muplus_particle_ref": [101, 201],
            "muminus_particle_ref": [102, 202],
            "muplus_d0": [0.2, 0.0],
            "muminus_d0": [-0.3, 0.0],
            "muplus_error_d0": [0.1, 0.0],
            "muminus_error_d0": [0.1, 0.0],
            "dimuon_px": [3.0, 1.0],
            "dimuon_py": [4.0, 0.0],
            "dimuon_pz": [0.0, 1.0],
            "dimuon_mass": [2.0, 1.0],
            "sv_x": [6.0, 0.0],
            "sv_y": [8.0, 0.0],
            "sv_z": [0.0, 0.0],
        })
        self.particles = pd.DataFrame([
            {
                "event_id": 1,
                "object_index": 0,
                "root_uid": 100,
                "PID": 32,
                "M1": -1,
                "M2": -1,
                "Mass": 2.0,
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "PX": 3.0,
                "PY": 4.0,
                "PZ": 0.0
            },
            {
                "event_id": 1,
                "object_index": 1,
                "root_uid": 101,
                "PID": -13,
                "M1": 0,
                "M2": 0,
                "Mass": delphes.MUON_MASS_GEV,
                "X": 6.0,
                "Y": 8.0,
                "Z": 0.0,
                "PX": 1.0,
                "PY": 2.0,
                "PZ": 0.0
            },
            {
                "event_id": 1,
                "object_index": 2,
                "root_uid": 102,
                "PID": 13,
                "M1": 0,
                "M2": 0,
                "Mass": delphes.MUON_MASS_GEV,
                "X": 6.0,
                "Y": 8.0,
                "Z": 0.0,
                "PX": 2.0,
                "PY": 2.0,
                "PZ": 0.0
            },
            {
                "event_id": 2,
                "object_index": 0,
                "root_uid": 200,
                "PID": 22,
                "M1": -1,
                "M2": -1,
                "Mass": 0.0,
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "PX": 1.0,
                "PY": 0.0,
                "PZ": 0.0
            },
            {
                "event_id": 2,
                "object_index": 1,
                "root_uid": 201,
                "PID": -13,
                "M1": 0,
                "M2": 0,
                "Mass": delphes.MUON_MASS_GEV,
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "PX": 0.5,
                "PY": 0.0,
                "PZ": 0.0
            },
            {
                "event_id": 2,
                "object_index": 2,
                "root_uid": 202,
                "PID": 13,
                "M1": 0,
                "M2": 0,
                "Mass": delphes.MUON_MASS_GEV,
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "PX": 0.5,
                "PY": 0.0,
                "PZ": 0.0
            },
        ])

    def test_add_ip_proxies_mark_zero_errors_as_invalid(self) -> None:
        result = delphes.add_ip_proxies(self.dimuons)
        self.assertAlmostEqual(result.loc[0, "sqrt_min_chi2_ip_proxy"], 2.0)
        self.assertAlmostEqual(result.loc[0, "min_chi2_ip_proxy"], 4.0)
        self.assertTrue(result.loc[0, "ip_proxy_valid"])
        self.assertTrue(np.isnan(result.loc[1, "min_chi2_ip_proxy"]))
        self.assertFalse(result.loc[1, "ip_proxy_valid"])

    def test_existing_helpers_accept_mass_column_and_signed_charge(self) -> None:
        muons = pd.DataFrame({
            "event_id": [1, 1],
            "PT": [3.0, 4.0],
            "Eta": [0.0, 0.0],
            "Phi": [0.0, np.pi/2],
            "Mass": [1.0, 2.0],
            "Charge": [1, -1]
        })
        momenta = delphes.calc_four_momentum(muons, mass="Mass")
        negative = delphes.select_muons(muons, charge=-1)
        self.assertAlmostEqual(momenta.loc[0, "E"], np.sqrt(10.0))
        self.assertEqual(negative["Charge"].tolist(), [-1])

    def test_invalid_ip_is_not_classified_as_mixed(self) -> None:
        result = delphes.classify_dimuon_displacement(self.dimuons)
        self.assertEqual(result.loc[1, "displacement_category"], "invalid")
    def test_truth_decay_time_uses_common_parent(self) -> None:
        result = delphes.calc_truth_decay_time(self.dimuons, self.particles, parent_pid=32)
        expected_time = 4.0/delphes.C_MM_PS
        self.assertEqual(result.loc[0, "truth_parent_pid"], 32)
        self.assertAlmostEqual(result.loc[0, "truth_decay_length_mm"], 10.0)
        self.assertAlmostEqual(result.loc[0, "truth_ctau_mm"], 4.0)
        self.assertAlmostEqual(result.loc[0, "dimuon_decay_time_truth_ps"], expected_time)
        self.assertTrue(np.isnan(result.loc[1, "dimuon_decay_time_truth_ps"]))
        summary = delphes.check_long_lived_truth(result)
        self.assertTrue(summary["has_long_lived_truth"])
        self.assertEqual(summary["n_long_lived_truth"], 1)
    def test_missing_truth_parent_is_reported_as_not_found(self) -> None:
        particles = self.particles.copy()
        particles.loc[
            (particles["event_id"] == 1) & particles["PID"].abs().eq(13),
            ["M1", "M2"]
        ] = 99
        result = delphes.calc_truth_decay_time(
            self.dimuons.iloc[[0]],
            particles,
            parent_pid=32
        )
        self.assertFalse(result.iloc[0]["truth_parent_found"])
        self.assertTrue(np.isnan(result.iloc[0]["dimuon_decay_time_truth_ps"]))
    def test_reco_decay_time_uses_parent_mass_and_momentum(self) -> None:
        result = delphes.calc_reco_decay_time(self.dimuons, primary_vertex=(0.0, 0.0, 0.0))
        self.assertAlmostEqual(result.loc[0, "dimuon_ctau_reco_mm"], 4.0)
        self.assertAlmostEqual(result.loc[0, "dimuon_decay_time_reco_ps"], 4.0/delphes.C_MM_PS)
    def test_decay_fit_proxy_is_deterministic(self) -> None:
        candidate = self.dimuons.iloc[[0]].assign(
            muplus_track_x = 6.01,
            muplus_track_y = 8.0,
            muplus_track_z = 0.2,
            muminus_track_x = 5.99,
            muminus_track_y = 8.0,
            muminus_track_z = 0.2,
        )
        result = delphes.calc_decay_fit_proxy(
            candidate,
            vertex_xy_resolution_mm=0.01,
            vertex_z_resolution_mm=0.2,
            pointing_resolution_mm=0.1,
            primary_vertex=(0.0, 0.0, 0.0)
        )
        self.assertAlmostEqual(result.iloc[0]["dimuon_vertex_separation_xy_mm"], 0.02)
        self.assertAlmostEqual(result.iloc[0]["dimuon_pointing_miss_mm"], 0.0)
        self.assertAlmostEqual(result.iloc[0]["chi2_df_proxy"], 8.0)
    def test_track_vertices_are_matched_by_particle_reference(self) -> None:
        tracks = pd.DataFrame({
            "event_id": [1, 1],
            "Particle_ref": [101, 102],
            "object_index": [4, 5],
            "X": [6.0, 6.0],
            "Y": [8.0, 8.0],
            "Z": [0.1, -0.1],
        })
        result = delphes.attach_dimuon_track_vertices(self.dimuons.iloc[[0]], tracks)
        self.assertTrue(result.iloc[0]["dimuon_tracks_matched"])
        self.assertEqual(result.iloc[0]["muplus_track_index"], 4)
        self.assertEqual(result.iloc[0]["muminus_track_index"], 5)

if __name__ == "__main__":
    unittest.main()