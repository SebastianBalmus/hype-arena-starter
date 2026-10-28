import unittest

from knowledge_base.parser import AssertionParser


LEX = {"subjects": [
    {"name": "Hooly", "aliases": ["Hooly AI"], "attributes": ["price"]},
    {"name": "Hooly AI Pro", "aliases": ["Pro"], "attributes": ["price"]},
    {"name": "Orbit", "aliases": [], "attributes": ["users"]},
]}
TRAIN = [
    {"text": "Hooly AI Pro costs €49 a month.", "form": [{"subject": "Hooly AI Pro", "attribute": "price", "value": {"kind": "money", "value": 49, "currency": "EUR", "per": "month"}}]},
    {"text": "Orbit serves 40 users.", "form": [{"subject": "Orbit", "attribute": "users", "value": {"kind": "quantity", "value": 40, "unit": "user"}}]},
    # All labels may teach parsing; forms are not world evidence.
    {"text": "Orbit serves 45 users.", "label": "false", "form": [{"subject": "Orbit", "attribute": "users", "value": {"kind": "quantity", "value": 45, "unit": "user"}}]},
]


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = AssertionParser(TRAIN, LEX)

    def test_longest_alias_and_changed_numeric_surface(self):
        got = self.parser.parse("Hooly AI Pro costs €59 a month.")
        self.assertTrue(got["complete"])
        self.assertEqual(got["atoms"][0]["subject"], "Hooly AI Pro")
        self.assertEqual(got["atoms"][0]["attribute"], "price")
        self.assertEqual(got["atoms"][0]["value"], {"kind": "money", "value": "59", "currency": "EUR", "per": "month"})

    def test_quantity_and_all_label_template_training(self):
        got = self.parser.parse("Orbit serves 52 users.")
        self.assertTrue(got["complete"])
        self.assertEqual(got["atoms"][0]["value"]["value"], "52")
        self.assertEqual(self.parser.summary()["eligible_rows"], 3)

    def test_extra_clause_rejected(self):
        self.assertFalse(self.parser.parse("Orbit serves 52 users and costs €8.")["complete"])
        self.assertFalse(self.parser.parse("Orbit serves 52 users and offers encryption.")["complete"])

    def test_operator_is_preserved_and_unseen_operator_template_not_assumed(self):
        # A strict whole-template learner does not generalize a changed operator wording.
        got = self.parser.parse("Orbit serves over 40 users.")
        self.assertFalse(got["complete"])

    def test_date_and_negation_rejected(self):
        self.assertFalse(self.parser.parse("Orbit served 40 users in 2025.")["complete"])
        self.assertFalse(self.parser.parse("Orbit does not serve 40 users.")["complete"])
        self.assertFalse(self.parser.parse("Orbit can't serve 40 users.")["complete"])
        self.assertFalse(self.parser.parse("Orbit expects 40 users next quarter.")["complete"])
        self.assertFalse(self.parser.parse("Orbit targets 40 users.")["complete"])

    def test_ambiguous_interpretations(self):
        rows = TRAIN + [{"text": "Orbit serves 40 users.", "form": [{"subject": "Orbit", "attribute": "price", "value": {"kind": "quantity", "value": 40, "unit": "user"}}]}]
        lex = {"subjects": [dict(s, attributes=s["attributes"] + (["price"] if s["name"] == "Orbit" else [])) for s in LEX["subjects"]]}
        p = AssertionParser(rows, lex)
        self.assertEqual(p.parse("Orbit serves 50 users.")["reason"], "ambiguous_parse")

    def test_same_template_for_different_subjects_is_not_attribute_conflict(self):
        lex = {"subjects": [
            {"name": "Orbit", "aliases": [], "attributes": ["users"]},
            {"name": "Nova", "aliases": [], "attributes": ["users"]},
        ]}
        rows = [
            {"text": "Orbit serves 40 users.", "form": [{"subject": "Orbit", "attribute": "users", "value": {"kind": "quantity", "value": 40, "unit": "user"}}]},
            {"text": "Nova serves 40 users.", "form": [{"subject": "Nova", "attribute": "users", "value": {"kind": "quantity", "value": 40, "unit": "user"}}]},
        ]
        p = AssertionParser(rows, lex)
        self.assertTrue(p.parse("Orbit serves 52 users.")["complete"])

    def test_sentence_final_decimal_and_periodic_price(self):
        training = [{"text": "Hooly AI Pro costs €49 per month.", "form": [{"subject": "Hooly AI Pro", "attribute": "price", "value": {"kind": "money", "value": 49, "currency": "EUR", "per": "month"}}]}]
        p = AssertionParser(training, LEX)
        parsed = p.parse("Hooly AI Pro costs €59 per month.")
        self.assertTrue(parsed["complete"])
        self.assertEqual(parsed["atoms"][0]["value"]["per"], "month")

    def test_operator_rejected_when_not_training_supported(self):
        self.assertFalse(self.parser.parse("Orbit serves over 40 users.")["complete"])


if __name__ == "__main__":
    unittest.main()
