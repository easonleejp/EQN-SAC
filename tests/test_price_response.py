"""Price and response diagnostics tested against exogenous, hand-worked cases."""

import copy
import unittest

import numpy as np

from tools.analyze_price_response import analyze_case, analyze_evaluation, correlation, quartile_masks


class PriceResponseTests(unittest.TestCase):
    def setUp(self):
        self.environment = dict(
            params=dict(dt=1),
            storage=dict(soc_min=0, soc_max=10, soc_init=0, eta_ch=1, eta_dis=1,
                         eta_self=1, p_ch_max=8, p_dis_max=4, c_es=.1),
            user=dict(price_min=[10] * 3, price_max=[25] * 3),
        )
        self.profile = dict(retail_price=[10, 10, 20, 20], grid_price=[2, 1, 4, 3],
                            baseline_load=[10, 20, 30, 40])
        self.case = dict(case=0, scenario_seed=10, response_seed=20,
                         levels=["low", "medium", "high"], season="winter", hourly=[])
        values = dict(baseline_load=[10, 20, 30, 40], wind=[2] * 4, grid_price=[2, 1, 4, 3],
                      user_load=[12, 21, 25, 35], price=[12, 11, 22, 21],
                      charge=[2, 0, 0, 0], discharge=[0, 0, 1, 1],
                      initial_soc=[0, 2, 2, 1], soc=[2, 2, 1, 0], grid_purchase=[12, 19, 22, 32])
        for hour in range(4):
            self.case["hourly"].append(dict(hour=hour, **{key: value[hour] for key, value in values.items()}))

    def test_price_alignment_uses_original_tou_and_actual_wholesale(self):
        result = analyze_case(self.case, self.environment, self.profile)
        self.assertEqual(result["prices"]["correlation_with_realized_wholesale"]["spearman"], 1)
        self.assertAlmostEqual(result["prices"]["tou_bands"]["10.0"]["retail_price_mean"], 11.5)
        self.assertAlmostEqual(result["prices"]["tou_bands"]["20.0"]["retail_price_mean"], 21.5)
        self.assertEqual(result["prices"]["tou_high_minus_low_mean"], 10)
        self.assertTrue(result["prices"]["tou_means_nondecreasing"])
        self.assertEqual(result["bands"]["nominal_wholesale"]["low"]["hours"], [1])
        self.assertEqual(result["bands"]["nominal_wholesale"]["high"]["hours"], [2])

    def test_actual_power_and_weighted_buy_sell_prices(self):
        # Deliberately contradictory raw actions must not influence physical diagnosis.
        for row in self.case["hourly"]:
            row["raw_storage_action"] = -1
            row["executed_storage_action"] = -1
        result = analyze_case(self.case, self.environment, self.profile)
        storage = result["storage"]
        self.assertEqual(storage["charge_energy"], 2)
        self.assertEqual(storage["discharge_energy"], 2)
        self.assertEqual(storage["weighted_charge_wholesale_price"], 2)
        self.assertEqual(storage["weighted_discharge_wholesale_price"], 3.5)
        self.assertEqual(storage["discharge_minus_charge_price"], 1.5)
        self.assertEqual(storage["realized_wholesale_low_charge_fraction"], 0)
        self.assertEqual(storage["realized_wholesale_high_discharge_fraction"], .5)
        self.assertEqual(storage["same_price_grid_cost_saving"], 3)
        self.assertAlmostEqual(storage["throughput_cost"], .4)
        self.assertAlmostEqual(storage["same_price_storage_margin"], 2.6)

    def test_original_low_high_demand_and_no_storage_decomposition(self):
        result = analyze_case(self.case, self.environment, self.profile)
        self.assertTrue(result["checks"]["all_ok"])
        low = result["bands"]["nominal_load"]["low"]
        high = result["bands"]["nominal_load"]["high"]
        self.assertEqual(low["response_reduction_mean"], -2)
        self.assertEqual(high["response_reduction_mean"], 5)
        self.assertEqual(low["demand_grid_effect_mean"], 2)
        self.assertEqual(low["storage_grid_effect_mean"], 2)
        self.assertEqual(low["total_grid_effect_mean"], 4)
        self.assertEqual(high["demand_grid_effect_mean"], -5)
        self.assertEqual(high["storage_grid_effect_mean"], -1)
        self.assertEqual(high["total_grid_effect_mean"], -6)
        self.assertEqual(result["response"]["reduction_energy"], 7)
        self.assertEqual(result["grid_decomposition"]["max_additivity_error"], 0)
        np.testing.assert_array_equal([r["same_price_no_storage_grid"] for r in result["hourly"]], [10, 19, 23, 33])

    def test_grid_decomposition_preserves_zero_import_clipping(self):
        case = copy.deepcopy(self.case)
        for row in case["hourly"]:
            row["wind"] = 50
            row["grid_purchase"] = 0
        result = analyze_case(case, self.environment, self.profile)
        self.assertEqual(result["response"]["reduction_energy"], 7)
        self.assertEqual(result["grid_decomposition"]["total_effect_energy"], 0)
        self.assertEqual(result["grid_decomposition"]["max_additivity_error"], 0)
        self.assertAlmostEqual(result["storage"]["same_price_storage_margin"], -.4)

    def test_labels_do_not_change_when_policy_prices_change(self):
        first = analyze_case(self.case, self.environment, self.profile)
        other = copy.deepcopy(self.case)
        for row in other["hourly"]:
            row["price"] = 24 - row["hour"]
        second = analyze_case(other, self.environment, self.profile)
        for partition in ("nominal_wholesale", "realized_wholesale", "nominal_load", "realized_load"):
            for band in ("low", "middle", "high"):
                self.assertEqual(first["bands"][partition][band]["hours"], second["bands"][partition][band]["hours"])

    def test_case_distribution_does_not_claim_per_case_guarantee(self):
        reverse = copy.deepcopy(self.case)
        reverse["case"] = 1
        reverse["season"] = "summer"
        reverse["levels"] = dict(load="high", wind="low", price="medium")
        for row in reverse["hourly"]:
            row["price"] = 33 - row["price"]
        result = analyze_evaluation(dict(n=2, cases=[self.case, reverse]), self.environment, self.profile)
        summary = result["aggregate"]["metrics"]["prices.correlation_with_realized_wholesale.pearson"]
        self.assertAlmostEqual(summary["mean"], 0)
        self.assertEqual(summary["positive_fraction"], .5)
        self.assertEqual(summary["valid_cases"], 2)
        self.assertEqual(len(result["contexts"]), 2)
        self.assertEqual(result["context_coverage"]["explicit_season_cases"], 2)

    def test_missing_labels_and_constant_correlations_remain_unknown(self):
        case = copy.deepcopy(self.case)
        del case["season"], case["levels"]
        for row in case["hourly"]:
            row["price"] = 15
        result = analyze_evaluation(dict(cases=[case]), self.environment, self.profile)
        self.assertEqual(result["cases"][0]["context"]["season"], "unknown")
        metric = result["aggregate"]["metrics"]["prices.correlation_with_original_tou.pearson"]
        self.assertIsNone(metric["mean"])
        self.assertEqual(metric["valid_cases"], 0)
        self.assertEqual(result["context_coverage"]["explicit_all_source_levels_cases"], 0)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3])["pearson"])
        masks = quartile_masks([1, 1, 1, 1])
        self.assertFalse(masks["low"].any())
        self.assertFalse(masks["high"].any())
        self.assertTrue(masks["middle"].all())

    def test_no_throughput_has_no_arbitrage_price_spread(self):
        case = copy.deepcopy(self.case)
        for row in case["hourly"]:
            row.update(charge=0, discharge=0, initial_soc=0, soc=0,
                       grid_purchase=row["user_load"] - row["wind"])
        result = analyze_case(case, self.environment, self.profile)
        self.assertIsNone(result["storage"]["discharge_minus_charge_price"])
        self.assertIsNone(result["storage"]["weighted_charge_wholesale_price"])
        self.assertEqual(result["storage"]["same_price_storage_margin"], 0)

    def test_rejects_wrong_parameter_snapshots_and_incomplete_payload(self):
        payload = dict(n=1, cases=[self.case], metadata=dict(environment=self.environment, profile=self.profile))
        self.assertTrue(analyze_evaluation(payload, self.environment, self.profile)["all_physical_checks_passed"])
        wrong = copy.deepcopy(self.profile)
        wrong["retail_price"][0] += 1
        with self.assertRaisesRegex(ValueError, "metadata snapshot"):
            analyze_evaluation(payload, self.environment, wrong)
        with self.assertRaisesRegex(ValueError, "count"):
            analyze_evaluation(dict(n=2, cases=[self.case]), self.environment, self.profile)


if __name__ == "__main__":
    unittest.main()
