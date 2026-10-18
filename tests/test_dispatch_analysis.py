"""Hand-calculated dispatch examples, independent of learned policy behavior."""

import copy
import unittest

import numpy as np

from tools.analyze_dispatch import (
    analyze_case, analyze_evaluation, analyze_learning, curve_metrics, rollout_environment,
)


class DispatchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "params": {"dt": 1.0},
            "storage": {"soc_min": 0, "soc_max": 10, "soc_init": 2,
                        "eta_ch": .8, "eta_dis": .5, "eta_self": 1,
                        "p_ch_max": 5, "p_dis_max": 5},
            "user": {"price_min": [10, 10, 10], "price_max": [20, 20, 20]},
        }
        self.case = {
            "case": 0,
            "hourly": [
                dict(hour=0, baseline_load=10, wind=2, user_load=8, charge=2,
                     discharge=0, initial_soc=2, soc=3.6, price=10, grid_purchase=8,
                     total_response=2, profit=7),
                dict(hour=1, baseline_load=20, wind=4, user_load=16, charge=0,
                     discharge=1, initial_soc=3.6, soc=1.6, price=20, grid_purchase=11,
                     total_response=4, profit=8),
            ],
            "reference_hourly": [
                dict(hour=0, user_load=10, charge=0, discharge=0, soc=2, price=15, profit=5),
                dict(hour=1, user_load=20, charge=0, discharge=0, soc=2, price=15, profit=6),
            ],
        }

    def test_curve_statistics_use_population_and_adjacent_ramps(self):
        stats = curve_metrics([2, 6, 4], dt_hours=.5)
        self.assertEqual(stats["peak"], 6)
        self.assertEqual(stats["valley"], 2)
        self.assertEqual(stats["peak_valley"], 4)
        self.assertEqual(stats["par"], 1.5)
        self.assertEqual(stats["energy"], 6)
        self.assertAlmostEqual(stats["std"], np.sqrt(8 / 3))
        self.assertEqual(stats["mean_abs_ramp"], 6)
        self.assertAlmostEqual(stats["rms_ramp"], np.sqrt(40))
        self.assertEqual(stats["max_abs_ramp"], 8)
        self.assertIsNone(curve_metrics([0, 0])["par"])
        self.assertEqual(curve_metrics([3])["rms_ramp"], 0)

    def test_hand_calculated_paired_load_and_energy_metrics(self):
        result = analyze_case(self.case, self.spec)
        grid = result["curves"]["grid"]
        self.assertEqual(grid["peak"], 11)
        self.assertEqual(grid["valley"], 8)
        self.assertEqual(grid["mean"], 9.5)
        self.assertAlmostEqual(grid["par"], 11 / 9.5)
        self.assertEqual(grid["std"], 1.5)
        self.assertEqual(grid["energy"], 19)
        self.assertEqual(result["curves"]["baseline_load"]["energy"], 30)
        self.assertEqual(result["curves"]["user_load"]["energy"], 24)
        self.assertEqual(result["curves"]["baseline_grid"]["energy"], 24)
        self.assertEqual(result["curves"]["response_grid_without_storage"]["energy"], 18)
        storage = result["storage"]
        self.assertEqual(storage["charge_energy"], 2)
        self.assertEqual(storage["discharge_energy"], 1)
        self.assertEqual(storage["bus_throughput"], 3)
        self.assertAlmostEqual(storage["cell_throughput"], 3.6)
        self.assertAlmostEqual(storage["equivalent_full_cycles"], .18)
        self.assertAlmostEqual(storage["conversion_losses"], 1.4)
        self.assertAlmostEqual(storage["soc_change"], -.4)
        self.assertAlmostEqual(storage["charge_energy"] - storage["discharge_energy"] - storage["conversion_losses"], storage["soc_change"])
        self.assertTrue(result["checks"]["all_ok"])
        self.assertTrue(result["reference"]["checks"]["all_ok"])
        self.assertEqual(result["paired_comparison"]["peak_reduction"], 5)
        self.assertEqual(result["paired_comparison"]["valley_lift"], 0)
        self.assertEqual(result["paired_comparison"]["std_reduction"], 2.5)
        self.assertEqual(result["paired_comparison"]["profit_change"], 4)
        self.assertEqual(result["paired_comparison"]["grid_energy_change"], -5)
        self.assertEqual(result["prices"]["lower_fraction"], .5)
        self.assertEqual(result["prices"]["upper_fraction"], .5)

    def test_surplus_is_not_export_and_balances_energy(self):
        case = {"hourly": [dict(hour=0, baseline_load=2, wind=10, user_load=2,
                                charge=0, discharge=1, soc=0, initial_soc=2,
                                price=15, grid_purchase=0)]}
        result = analyze_case(case, self.spec)
        self.assertEqual(result["curves"]["grid"]["energy"], 0)
        self.assertEqual(result["storage"]["surplus_energy"], 9)
        # Surplus includes 1 unit of battery discharge, so it is not all curtailed wind.
        self.assertEqual(10 + 1, 2 + result["storage"]["surplus_energy"])
        self.assertTrue(result["checks"]["all_ok"])

    def test_detects_inconsistent_recorded_physics(self):
        broken = copy.deepcopy(self.case)
        broken["hourly"][1]["grid_purchase"] += 1
        broken["hourly"][1]["soc"] += .1
        broken["hourly"][1]["initial_soc"] += .2
        result = analyze_case(broken, self.spec)
        self.assertFalse(result["checks"]["grid_balance_ok"])
        self.assertFalse(result["checks"]["soc_balance_ok"])
        self.assertFalse(result["checks"]["initial_soc_chain_ok"])
        self.assertFalse(result["checks"]["all_ok"])
        self.assertAlmostEqual(result["checks"]["soc_balance_max_abs_error"], .1)

    def test_self_discharge_and_half_hour_steps(self):
        spec = copy.deepcopy(self.spec)
        spec["params"]["dt"] = .5
        spec["storage"]["eta_self"] = .9
        case = {"hourly": [dict(hour=0, baseline_load=2, wind=0, user_load=2,
                                charge=2, discharge=0, soc=2.6, initial_soc=2,
                                price=15, grid_purchase=4)]}
        result = analyze_case(case, spec)
        self.assertEqual(result["curves"]["grid"]["energy"], 2)
        self.assertEqual(result["storage"]["charge_energy"], 1)
        self.assertAlmostEqual(result["storage"]["self_discharge_losses"], .2)
        self.assertTrue(result["checks"]["all_ok"])

    def test_rejects_nonfinite_malformed_or_unpaired_data(self):
        for field, value in (("wind", float("nan")), ("charge", -1), ("hour", 5)):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.case)
                broken["hourly"][0][field] = value
                with self.assertRaises(ValueError):
                    analyze_case(broken, self.spec)
        broken = copy.deepcopy(self.case)
        broken["reference_hourly"][0]["wind"] = 9
        broken["reference_hourly"][1]["wind"] = 4
        with self.assertRaisesRegex(ValueError, "wind differ"):
            analyze_case(broken, self.spec)
        broken = copy.deepcopy(self.case)
        del broken["hourly"][0]["grid_purchase"]
        with self.assertRaisesRegex(ValueError, "every row or none"):
            analyze_case(broken, self.spec)

    def test_evaluation_aggregates_daily_metrics_not_average_curve_peaks(self):
        other = copy.deepcopy(self.case)
        other["case"] = 1
        result = analyze_evaluation({"cases": [self.case, other]}, self.spec)
        self.assertEqual(result["n"], 2)
        self.assertTrue(result["checks_passed"])
        self.assertEqual(result["aggregate"]["curves.grid.peak"], {"mean": 11, "std": 0, "n": 2})
        self.assertEqual(result["aggregate"]["paired_comparison.profit_change"]["mean"], 4)

    def test_learning_windows_preserve_missing_metrics_and_sampled_semantics(self):
        rows = [dict(episode=i, profit=2*i, reference_profit=i, critic_loss="" if i == 1 else i,
                     seconds=i, updates=max(i-1, 0)) for i in (1, 2, 3)]
        result = analyze_learning(rows, window=2)
        self.assertEqual(result["rows"][-1]["rolling_profit_change"], 2.5)
        self.assertIsNone(result["rows"][0]["rolling_critic_loss"])
        self.assertEqual(result["rows"][1]["rolling_critic_loss"], 2)
        self.assertAlmostEqual(result["summary"]["profit"]["last_window_slope_per_episode"], 2)
        self.assertEqual(result["summary"]["critic_loss"]["available_count"], 2)
        with self.assertRaises(ValueError):
            analyze_learning([rows[1], rows[0]])

    def test_fixed_environment_probe_uses_physical_zero_storage(self):
        from eqnsac.environment import Environment
        env = Environment()
        case = env.generator.sample(123)
        trajectory = rollout_environment(env, case, lambda state: [0, 0])
        self.assertEqual(len(trajectory["hourly"]), 24)
        result = analyze_case(trajectory, env.spec)
        self.assertTrue(result["checks"]["all_ok"])
        self.assertEqual(result["storage"]["bus_throughput"], 0)
        self.assertEqual(result["storage"]["initial_soc"], result["storage"]["final_soc"])


if __name__ == "__main__":
    unittest.main()
