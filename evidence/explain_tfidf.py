"""Rebuild the fitted sparse baseline and audit additive class margins in selected errors."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "review_run_1"
DATA = ROOT / "data" / "processed" / "qevasion_grouped.csv"
PRED = RUN / "predictions" / "grouped_tfidf.csv"
CASES = ("qevasion-test-0119", "qevasion-test-0240", "qevasion-test-0000")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def answer_segments(case_id, answer):
    if case_id == "qevasion-test-0119":
        boundary = answer.rindex("Yes.")
        return [("delegation_statement", answer[:boundary]), ("closing_yes", answer[boundary:])]
    if case_id == "qevasion-test-0240":
        refusal_end = answer.index(". I will comment") + 1
        policy_start = answer.index("The White House")
        return [("explicit_refusal", answer[:refusal_end]),
                ("pivot_to_other_comments", answer[refusal_end:policy_start]),
                ("fiscal_background_and_transcript", answer[policy_start:])]
    return [("answer", answer)]


def occurrence_segments(term, tokens, tags):
    gram = term.split()
    found = set()
    for start in range(len(tokens) - len(gram) + 1):
        if tokens[start:start + len(gram)] == gram:
            labels = set(tags[start:start + len(gram)])
            found.add(next(iter(labels)) if len(labels) == 1 else "cross_segment")
    if not found:
        raise AssertionError(f"Active feature cannot be located in the token stream: {term}")
    return sorted(found)


def explain(case_id, test, test_x, probabilities, logits, model, vectorizer, labels):
    i = test.index.get_loc(case_id)
    row = test.iloc[i]
    sample = test_x.getrow(i)
    prediction = labels[probabilities[i].argmax()]
    gold = row["label"]
    if prediction == gold:
        raise AssertionError(f"Audited case was not misclassified: {case_id}")
    classes = list(model.classes_)
    pred_code, gold_code = classes.index(prediction), classes.index(gold)
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[pred_code] - model.coef_[gold_code]
    intercept = float(model.intercept_[pred_code] - model.intercept_[gold_code])
    real_margin = float(logits[i, pred_code] - logits[i, gold_code])
    prob_log_margin = float(np.log(probabilities[i, labels.index(prediction)] / probabilities[i, labels.index(gold)]))

    tokenize = vectorizer.build_tokenizer()
    prepare = vectorizer.build_preprocessor()
    segments = [("question", row["question"]), ("pair_separator", " [SEP] "), *answer_segments(case_id, row["answer"])]
    exact = "".join(text for _, text in segments)
    tokens, tags = [], []
    for name, content in segments:
        part = tokenize(prepare(content))
        tokens.extend(part)
        tags.extend([name] * len(part))
    assert exact == row["question"] + " [SEP] " + row["answer"]
    assert tokens == tokenize(prepare(exact))

    terms = []
    segment_sums = {}
    for feature, weight in zip(sample.indices, sample.data):
        term = str(feature_names[feature])
        per_feature = float(weight * coefficients[feature])
        locations = occurrence_segments(term, tokens, tags)
        terms.append({"term": term, "normalized_tfidf": float(weight),
                      "coefficient_pred_minus_gold": float(coefficients[feature]), "margin_contribution": per_feature,
                      "found_in_segments": locations})
        location = locations[0] if len(locations) == 1 else "multiple_segments"
        contribution = segment_sums.setdefault(location, {"active_features": 0, "margin_contribution": 0.0})
        contribution["active_features"] += 1
        contribution["margin_contribution"] += per_feature
    total = sum(term["margin_contribution"] for term in terms)
    assert np.isclose(intercept + total, real_margin, atol=1e-12, rtol=0)
    assert np.isclose(prob_log_margin, real_margin, atol=1e-12, rtol=0)
    positive = sorted((term for term in terms if term["margin_contribution"] > 0), key=lambda t: -t["margin_contribution"])[:8]
    negative = sorted((term for term in terms if term["margin_contribution"] < 0), key=lambda t: t["margin_contribution"])[:8]
    focus_terms = {
        "qevasion-test-0119": {"yes", "so yes", "handling", "those conversations", "directly", "condi"},
        "qevasion-test-0240": {"not comment", "comment on", "decisions", "fiscal policy", "raise taxes", "budget", "taxes"},
    }.get(case_id, set())
    return {
        "id": case_id, "gold": gold, "prediction": prediction, "probabilities": dict(zip(labels, probabilities[i].tolist())),
        "input_locator": {"table_path": str(DATA.relative_to(ROOT).as_posix()), "split": "test", "id": case_id,
                          "input_sha256_utf8": sha256(exact.encode("utf-8")).hexdigest(),
                          "question_tokens_before_ngram": tags.count("question"),
                          "answer_tokens_before_ngram": len(tags) - tags.count("question") - tags.count("pair_separator")},
        "direction": f"{prediction} minus {gold}", "units": "multinomial class logit difference = natural log probability ratio",
        "intercept_pred_minus_gold": intercept, "all_feature_contributions_sum": total,
        "pred_minus_gold_margin": real_margin, "ln_probability_ratio": prob_log_margin,
        "additive_check_residual": real_margin - (intercept + total),
        "top_positive_toward_prediction": positive, "top_negative_toward_gold": negative,
        "other_feature_contributions_sum": total - sum(t["margin_contribution"] for t in positive + negative),
        "segment_contributions": segment_sums,
        "focus_feature_contributions": [term for term in terms if term["term"] in focus_terms],
    }


def main():
    source = pd.read_csv(DATA, keep_default_na=False, dtype={"id": str}).set_index("id", verify_integrity=True)
    train = source.loc[source["split"].eq("train")]
    test = source.loc[source["split"].eq("test")]
    config_path = RUN / "configuration.json"
    selection_path = RUN / "selection.json"
    config = read_json(config_path)
    tuning = read_json(selection_path)["grouped_tfidf"]
    selected = tuning["selected"]
    assert selected == max(tuning["candidates"], key=lambda candidate: candidate["validation_macro_f1"])

    def pair(frame):
        return frame["question"] + " [SEP] " + frame["answer"]

    vectorizer = TfidfVectorizer(ngram_range=(1, selected["ngram_top"]),
                                 min_df=config["tfidf_min_document_frequency"], max_features=config["tfidf_max_features"],
                                 sublinear_tf=True, token_pattern=r"(?u)\[sep\]|\b\w\w+\b")
    vectorizer.fit(pair(train).to_numpy())
    features = vectorizer.transform(pair(source).to_numpy())
    train_x = features[np.flatnonzero(source["split"].eq("train"))]
    test_x = features[np.flatnonzero(source["split"].eq("test"))]
    model = LogisticRegression(C=selected["c"], class_weight=selected["class_weight"], solver="lbfgs",
                               max_iter=config["max_lr_iterations"], random_state=config["seed"], tol=1e-5)
    with threadpool_limits(limits=config["threads"]):
        model.fit(train_x, train["label"])
        assert model.n_iter_.tolist() == selected["iterations"]
        assert test_x.shape[1] == selected["feature_count"]
        labels = config["source_labels"]
        probabilities = model.predict_proba(test_x)[:, [list(model.classes_).index(label) for label in labels]]
        logits = model.decision_function(test_x)

    saved = pd.read_csv(PRED, keep_default_na=False, dtype={"id": str}).set_index("id", verify_integrity=True)
    assert set(saved.index) == set(test.index)
    saved = saved.reindex(test.index)
    assert saved["gold"].eq(test["label"]).all() and saved["group"].eq(test["group"]).all()
    assert np.array_equal(np.asarray(labels)[probabilities.argmax(axis=1)], saved["prediction"].to_numpy())
    max_prob_abs_delta = float(np.max(np.abs(probabilities - saved[[f"prob_{label}" for label in labels]].to_numpy())))
    assert max_prob_abs_delta <= 5e-12, max_prob_abs_delta

    reports = [explain(case_id, test, test_x, probabilities, logits, model, vectorizer, labels) for case_id in CASES]
    payload = {
        "method": "Refit the validation-selected TF-IDF+multinomial LR using the saved split, then use weight × pairwise coefficient difference.",
        "model_input": "question + ' [SEP] ' + answer, with the original input characters recoverable by ID from the grouped CSV",
        "interpretation": "Segments partition additive features of the same fixed input. This does not estimate how the model would behave after editing the input.",
        "packages": {name: version(name) for name in ("scikit-learn", "numpy", "pandas")},
        "selected": selected, "fitted_training_n": len(train), "checked_test_n": len(test),
        "hard_predictions_identical": True, "max_probability_absolute_difference": max_prob_abs_delta,
        "sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in (DATA, PRED, config_path, selection_path)},
        "cases": reports,
    }
    output = ROOT / "evidence" / "tfidf_error_contributions.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Reproduced {len(test)} predictions; max abs probability difference={max_prob_abs_delta:.3g}")
    for entry in reports:
        print(f"{entry['id']} {entry['direction']}, margin={entry['pred_minus_gold_margin']:+.6f}")
        print("  +", [(t["term"], round(t["margin_contribution"], 6)) for t in entry["top_positive_toward_prediction"][:4]])
        print("  -", [(t["term"], round(t["margin_contribution"], 6)) for t in entry["top_negative_toward_gold"][:4]])
        print("  segments", entry["segment_contributions"])


if __name__ == "__main__":
    main()
