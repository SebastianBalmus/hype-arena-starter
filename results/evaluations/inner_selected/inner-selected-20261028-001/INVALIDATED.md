# Invalidated draft run

Invalidated: evaluator-only validation scoring wrote gold labels in the method input directory. The method runner did not read that file, but the v1 input-boundary rule requires separation. The original predictions and diagnostic scores are retained. The corrected run uses a new ID.
