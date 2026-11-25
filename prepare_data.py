"""Build fixed source and target partitions directly from upstream data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

import numpy as np
from pyarrow import parquet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent
SEED = 2026
NEAR_THRESHOLD = 0.90
SUBSTANTIAL_ANSWER = 16
SOURCE_FIELDS = (
    "id", "question", "answer", "label", "group", "source", "official_split", "split",
    "fine_label", "question_length", "answer_length", "component", "source_index",
    "title", "president", "date", "question_order", "group_candidates",
)
REGEX_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return " ".join(REGEX_WORD.findall(text))


def length(text):
    return len(normalize(text).split())


def present(value):
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows, fields=SOURCE_FIELDS):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_config():
    return json.loads((ROOT / "data_sources.json").read_text(encoding="utf-8"))


def ensure_inputs(raw_dir, specs, allow_download=True):
    files = []
    for filename, spec in specs.items():
        path = raw_dir / filename
        expected = spec["sha256"]
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(path)
            temp = raw_dir / (filename + ".part")
            errors = []
            for url in spec["download_urls"]:
                try:
                    request = Request(url, headers={"User-Agent": "coursework-nlp/1.0"})
                    with urlopen(request, timeout=30) as source, temp.open("wb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                    if sha256_file(temp) != expected:
                        raise ValueError("Downloaded bytes have an unexpected SHA-256")
                    temp.replace(path)
                    break
                except (OSError, ValueError) as error:
                    errors.append(f"{url}: {error}")
                    temp.unlink(missing_ok=True)
            else:
                raise RuntimeError("; ".join(errors))
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch: {path}, observed {observed}, expected {expected}")
        files.append({"file": filename, "bytes": path.stat().st_size,
                      "sha256": observed, "upstream_urls": spec["download_urls"]})
    return files


def canonical_url(url):
    parts = urlsplit(url.strip())
    if not parts.hostname:
        raise ValueError(f"Unusable interview URL: {url!r}")
    host = parts.hostname.removeprefix("www.")
    return host + unquote(parts.path).rstrip("/")


def qevasion_rows(raw_dir):
    all_rows = []
    fields = {}
    for split, expected in (("train", 3448), ("test", 308)):
        table = parquet.read_table(raw_dir / f"qevasion-{split}.parquet")
        raw = table.to_pylist()
        assert len(raw) == expected
        fields[split] = {
            "count": len(raw),
            "type": {item.name: str(item.type) for item in table.schema},
            "missing": {column: sum(not present(row[column]) for row in raw) for column in table.schema.names},
        }
        for offset, row in enumerate(raw):
            question = row["question"]
            answer = row["interview_answer"]
            label = row["clarity_label"]
            if not all(present(value) for value in (question, answer, label, row["url"])):
                raise ValueError(f"Missing core data in {split} row {offset}")
            all_rows.append({
                "id": f"qevasion-{split}-{offset:04d}",
                "question": question,
                "answer": answer,
                "label": label,
                "fine_label": row["evasion_label"],
                "group": canonical_url(row["url"]),
                "source": row["url"],
                "official_split": split,
                "question_length": length(question),
                "answer_length": length(answer),
                "source_index": row["index"],
                "title": row["title"],
                "president": row["president"],
                "date": row["date"],
                "question_order": row["question_order"],
            })
    return all_rows, fields


def pair_index(rows, columns):
    by_text = defaultdict(list)
    for pos, row in enumerate(rows):
        values = tuple(normalize(row[column]) for column in columns)
        if all(values):
            by_text[values].append(pos)
    return by_text


def exact_edges(index, kind):
    for key, members in index.items():
        if kind == "exact_answer_substantial" and len(key[0].split()) < SUBSTANTIAL_ANSWER:
            continue
        for left, right in combinations(members, 2):
            yield left, right, kind, None, None


def near_edges(rows, split_key=None):
    """Enumerate normalized char-4gram cosine matches for both question and answer."""
    strings = [normalize(row["question"]) for row in rows]
    answers = [normalize(row["answer"]) for row in rows]
    encoder = TfidfVectorizer(analyzer="char", ngram_range=(4, 4),
                              use_idf=False, binary=True, dtype=np.float32)
    question_vectors = encoder.fit_transform(strings).tocsr()
    answer_vectors = encoder.fit_transform(answers).tocsr()
    answer_transpose = answer_vectors.T.tocsr()
    question_transpose = question_vectors.T.tocsr()
    for start in range(0, len(rows), 128):
        stop = min(start + 128, len(rows))
        similarities = (question_vectors[start:stop] @ question_transpose).tocsr()
        for offset in range(stop - start):
            left = start + offset
            pos = slice(similarities.indptr[offset], similarities.indptr[offset + 1])
            other = similarities.indices[pos]
            scores = similarities.data[pos]
            enough = (other > left) & (scores >= NEAR_THRESHOLD)
            other = other[enough]
            scores = scores[enough]
            if split_key is not None:
                cross = np.array([rows[left][split_key] != rows[right][split_key] for right in other], dtype=bool)
                other, scores = other[cross], scores[cross]
            if not len(other):
                continue
            answer_scores = (answer_vectors[left] @ answer_transpose[:, other]).toarray()[0]
            for right, question_score, answer_score in zip(other, scores, answer_scores, strict=True):
                if answer_score < NEAR_THRESHOLD:
                    continue
                if strings[left] == strings[right] and answers[left] == answers[right]:
                    continue
                yield left, int(right), "near_question_and_answer", float(question_score), float(answer_score)


def components_for(rows, edges):
    roots = list(range(len(rows)))

    def root(pos):
        while roots[pos] != pos:
            roots[pos] = roots[roots[pos]]
            pos = roots[pos]
        return pos

    def join(left, right):
        a, b = sorted((root(left), root(right)))
        roots[b] = a

    for _, group in group_index(rows, "group").items():
        for right in group[1:]:
            join(group[0], right)
    for left, right, _, _, _ in edges:
        join(left, right)
    return [rows[root(pos)]["id"] for pos in range(len(rows))]


def group_index(rows, key):
    groups = defaultdict(list)
    for pos, row in enumerate(rows):
        groups[row[key]].append(pos)
    return groups


def pick_fold(rows, group_key):
    labels = [row["label"] for row in rows]
    groups = [row[group_key] for row in rows]
    totals = Counter(labels)
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(split.split(np.zeros(len(rows)), labels, groups=groups))

    def loss(fold):
        counts = Counter(labels[pos] for pos in fold[1])
        return sum(abs(counts[label] / total - 0.20) for label, total in totals.items())

    choice = min(range(len(folds)), key=lambda i: (loss(folds[i]), i))
    return set(folds[choice][1].tolist()), int(choice)


def length_summary(rows, field):
    values = np.array([row[field] for row in rows])
    low, q1, median, q3, p90, p95, high = np.quantile(values, [0, .25, .5, .75, .90, .95, 1])
    return {"mean": float(np.mean(values)), "min": float(low), "q1": float(q1),
            "median": float(median), "q3": float(q3), "p90": float(p90), "p95": float(p95), "max": float(high)}


def summarize(rows):
    result = {}
    for split in sorted({row["split"] for row in rows}):
        subset = [row for row in rows if row["split"] == split]
        counts = Counter(row["group"] for row in subset)
        result[split] = {
            "rows": len(subset),
            "labels": dict(sorted(Counter(row["label"] for row in subset).items())),
            "fine_labels": dict(sorted(Counter(row.get("fine_label") or "(missing)" for row in subset).items())),
            "source_group_count": len(counts),
            "duplicate_component_count": len({row["component"] for row in subset}),
            "source_group_size_median": float(np.median(list(counts.values()))),
            "source_group_size_max": int(max(counts.values())),
            "source_hosts": dict(sorted(Counter(row["group"].split("/", 1)[0] for row in subset).items())),
            "presidents": dict(sorted(Counter(row.get("president") or "(missing)" for row in subset).items())),
            "question_length": length_summary(subset, "question_length"),
            "answer_length": length_summary(subset, "answer_length"),
        }
    return result


def overlap(rows, split_key, group_key):
    split_groups = defaultdict(set)
    for row in rows:
        split_groups[row[split_key]].add(row[group_key])
    return {f"{left}~{right}": len(split_groups[left] & split_groups[right])
            for left, right in combinations(sorted(split_groups), 2)}


def duplicate_summary(index):
    repeated = [members for members in index.values() if len(members) > 1]
    return {"unique_duplicate_keys": len(repeated), "rows_in_duplicate_keys": sum(map(len, repeated)),
            "pair_count": sum(len(members) * (len(members) - 1) // 2 for members in repeated)}


def evidence_row(rows, edge):
    left, right, mode, qcos, acos = edge
    a, b = rows[left], rows[right]
    return {"id_a": a["id"], "id_b": b["id"], "kind": mode,
            "question_cosine": "" if qcos is None else f"{qcos:.6f}",
            "answer_cosine": "" if acos is None else f"{acos:.6f}",
            "split_a": a["split"], "split_b": b["split"],
            "official_split_a": a["official_split"], "official_split_b": b["official_split"],
            "source_a": a["source"], "source_b": b["source"],
            "label_a": a["label"], "label_b": b["label"]}


EDGE_FIELDS = ("id_a", "id_b", "kind", "question_cosine", "answer_cosine", "split_a", "split_b",
               "official_split_a", "official_split_b", "source_a", "source_b", "label_a", "label_b")


def prepare_source(raw_dir, out_dir):
    rows, fields = qevasion_rows(raw_dir)
    questions = pair_index(rows, ("question",))
    answers = pair_index(rows, ("answer",))
    qa_pairs = pair_index(rows, ("question", "answer"))
    edges = sorted([*exact_edges(qa_pairs, "exact_question_and_answer"),
                    *exact_edges(answers, "exact_answer_substantial"), *near_edges(rows)])
    for row, component in zip(rows, components_for(rows, edges), strict=True):
        row["component"] = component
    official_train = [row for row in rows if row["official_split"] == "train"]
    official_test = [row for row in rows if row["official_split"] == "test"]
    test_components = {row["component"] for row in official_test}
    test_sources = {row["group"] for row in official_test}
    val, selected_fold = pick_fold(official_train, "component")
    for pos, row in enumerate(official_train):
        row["split"] = "validation" if pos in val else "train"
    for row in official_test:
        row["split"] = "test"
    strict = [row for row in rows if row["official_split"] == "test" or row["component"] not in test_components]
    assert not any(overlap(strict, "split", "component").values())
    assert not any(overlap(strict, "split", "group").values())
    assert [row["id"] for row in strict if row["split"] == "test"] == [row["id"] for row in official_test]

    removed = [row for row in official_train if row["component"] in test_components]
    split_fields = ("id", "original_partition", "grouped_status", "reason", "group", "component")
    write_csv(out_dir / "qevasion_grouped_exclusions.csv", (
        {"id": row["id"], "original_partition": row["split"], "grouped_status": "excluded",
         "reason": "source_url_seen_in_official_test" if row["group"] in test_sources else "duplicate_component_link",
         "group": row["group"], "component": row["component"]} for row in removed), fields=split_fields)
    write_csv(out_dir / "qevasion_duplicate_links.csv", (evidence_row(rows, edge) for edge in edges), EDGE_FIELDS)
    write_csv(out_dir / "qevasion_official.csv", rows)
    write_csv(out_dir / "qevasion_grouped.csv", strict)
    group_rows = []
    for group, positions in sorted(group_index(rows, "group").items()):
        members = [rows[pos] for pos in positions]
        group_rows.append({"group": group, "source": members[0]["source"],
                           "official_train": sum(row["official_split"] == "train" for row in members),
                           "official_test": sum(row["official_split"] == "test" for row in members),
                           "grouped_excluded": sum(row["official_split"] == "train" and row["component"] in test_components for row in members)})
    write_csv(out_dir / "qevasion_source_groups.csv", group_rows, tuple(group_rows[0]))

    raw_summary = summarize([dict(row, split=row["official_split"]) for row in rows])
    cross_edges = [(p, q, kind) for p, q, kind, _, _ in edges if rows[p]["official_split"] != rows[q]["official_split"]]
    strict_ids = {row["id"] for row in strict}
    assert not any(rows[p]["id"] in strict_ids and rows[q]["id"] in strict_ids and rows[p]["split"] != rows[q]["split"]
                   for p, q, *_ in edges)
    train_sources = {row["group"] for row in official_train}
    audit = {
        "raw_fields": fields,
        "raw_splits": raw_summary,
        "official_protocol": summarize(rows),
        "grouped_protocol": summarize(strict),
        "raw_source_groups_total": len(group_rows),
        "raw_source_group_overlap": overlap(rows, "official_split", "group"),
        "raw_duplicate_component_overlap": overlap(rows, "official_split", "component"),
        "official_partition_source_overlap": overlap(rows, "split", "group"),
        "grouped_partition_source_overlap": overlap(strict, "split", "group"),
        "grouped_partition_component_overlap": overlap(strict, "split", "component"),
        "official_test_rows_from_overlapping_source": sum(row["group"] in train_sources for row in official_test),
        "strict_holdout_excluded": len(removed),
        "strict_holdout_excluded_for_source_overlap": sum(row["group"] in test_sources for row in removed),
        "strict_holdout_excluded_for_duplicate_link_only": sum(row["group"] not in test_sources for row in removed),
        "strict_holdout_excluded_by_partition": dict(Counter(row["split"] for row in removed)),
        "seed": SEED,
        "stratified_group_fold_zero_indexed": selected_fold,
        "duplicates": {"question": duplicate_summary(questions), "answer": duplicate_summary(answers),
                       "question_and_answer": duplicate_summary(qa_pairs),
                       "link_totals": dict(sorted(Counter(edge[2] for edge in edges).items())),
                       "cross_official_split_links": dict(sorted(Counter(edge[2] for edge in cross_edges).items())),
                       "cross_grouped_partition_links": 0,
                       "clashing_exact_qa_keys": sum(len({rows[pos]["label"] for pos in members}) > 1 for members in qa_pairs.values())},
        "fine_by_clarity_train": {label: dict(sorted(Counter(row["fine_label"] for row in official_train if row["label"] == label).items()))
                                  for label in sorted({row["label"] for row in official_train})},
    }
    write_json(out_dir / "qevasion_stats.json", audit)
    print(f"QEvasion official: {Counter(row['split'] for row in rows)}", flush=True)
    print(f"QEvasion grouped: {Counter(row['split'] for row in strict)}", flush=True)
    return rows, audit


def squad_articles(raw_dir):
    by_question = defaultdict(set)
    for split in ("train", "dev"):
        path = raw_dir / f"squad-{split}-v1.1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == "1.1"
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    by_question[normalize(qa["question"])].add(article["title"])
    return by_question


def article_components(by_question):
    roots = {title: title for titles in by_question.values() for title in titles}

    def root(title):
        while roots[title] != title:
            roots[title] = roots[roots[title]]
            title = roots[title]
        return title

    for titles in by_question.values():
        ordered = sorted(titles)
        for title in ordered[1:]:
            a, b = sorted((root(ordered[0]), root(title)))
            roots[b] = a
    return {title: root(title) for title in roots}


def qnli_rows(raw_dir, by_question, roots):
    result = {}
    with zipfile.ZipFile(raw_dir / "QNLI.zip") as archive:
        for split, expected in (("train", 104743), ("dev", 5463), ("test", 5463)):
            with archive.open(f"QNLI/{split}.tsv") as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", newline=""),
                                        delimiter="\t", quoting=csv.QUOTE_NONE)
                assert reader.fieldnames == ["index", "question", "sentence"] + ([] if split == "test" else ["label"])
                rows = list(reader)
            assert len(rows) == expected and len({row["index"] for row in rows}) == expected
            assert all(None not in row and all(present(value) for value in row.values()) for row in rows)
            if split == "test":
                result[split] = {"rows": len(rows), "labeled": False, "fields": reader.fieldnames}
                continue
            result[split] = []
            for row in rows:
                question, answer = row["question"], row["sentence"]
                normalized = normalize(question)
                articles = sorted(by_question.get(normalized, set()))
                source = "|".join(articles)
                group = f"wiki:{roots[articles[0]]}" if articles else f"unresolved_question:{normalized}"
                assert len({roots[article] for article in articles}) <= 1
                result[split].append({
                    "id": f"qnli-{split}-{int(row['index']):06d}",
                    "question": question, "answer": answer, "label": row["label"], "group": group,
                    "source": source, "official_split": split, "fine_label": "", "source_index": row["index"],
                    "question_length": length(question), "answer_length": length(answer),
                    "component": group, "group_candidates": len(articles),
                })
    return result


def sample_rows(rows, size, seed):
    generator = np.random.default_rng(seed)
    choice = sorted(generator.choice(len(rows), min(size, len(rows)), replace=False))
    return [rows[pos] for pos in choice]


def prepare_target(raw_dir, out_dir):
    by_question = squad_articles(raw_dir)
    data = qnli_rows(raw_dir, by_question, article_components(by_question))
    raw_train, test = data["train"], data["dev"]
    assert {row["label"] for row in raw_train} == {"entailment", "not_entailment"}
    test_groups = {row["group"] for row in test}
    test_questions = {normalize(row["question"]) for row in test}
    eligible = [row for row in raw_train if row["source"] and row["group"] not in test_groups
                and normalize(row["question"]) not in test_questions]
    val_indices, fold = pick_fold(eligible, "group")
    train_pool = [row for i, row in enumerate(eligible) if i not in val_indices]
    val_pool = [row for i, row in enumerate(eligible) if i in val_indices]
    train = sample_rows(train_pool, 12000, SEED + 1)
    val = sample_rows(val_pool, 2000, SEED + 2)
    for part, partition in ((train, "train"), (val, "validation"), (test, "test")):
        for row in part:
            row["split"] = partition
    rows = train + val + test

    edges = []
    for kind, index in (("exact_question_and_answer", pair_index(rows, ("question", "answer"))),
                        ("exact_answer_substantial", pair_index(rows, ("answer",)))):
        edges.extend(edge for edge in exact_edges(index, kind) if rows[edge[0]]["split"] != rows[edge[1]]["split"])
    edges.extend(near_edges(rows, split_key="split"))
    order = {"train": 0, "validation": 1, "test": 2}
    excluded = set()
    for left, right, *_ in edges:
        removed = min((left, right), key=lambda position: order[rows[position]["split"]])
        excluded.add(removed)
    kept = [row for pos, row in enumerate(rows) if pos not in excluded]
    assert not any(overlap(kept, "split", "group").values())
    assert [row for row in kept if row["split"] == "test"] == test
    assert not any(left not in excluded and right not in excluded for left, right, *_ in edges)

    write_csv(out_dir / "qnli.csv", kept)
    write_csv(out_dir / "qnli_duplicate_links.csv", (evidence_row(rows, edge) for edge in sorted(edges)), EDGE_FIELDS)
    write_csv(out_dir / "qnli_exclusions.csv", ({"id": rows[pos]["id"], "split": rows[pos]["split"],
                                                "group": rows[pos]["group"]} for pos in sorted(excluded)),
              ("id", "split", "group"))
    raw_summary = {}
    for split in ("train", "dev"):
        raw_part = data[split]
        raw_summary[split] = {
            "rows": len(raw_part), "fields": ["index", "question", "sentence", "label"],
            "labels": dict(sorted(Counter(row["label"] for row in raw_part).items())),
            "missing_values": {column: 0 for column in ("index", "question", "sentence", "label")},
            "article_join_coverage": dict(sorted(Counter(str(row["group_candidates"]) for row in raw_part).items())),
            "source_group_count": len({row["group"] for row in raw_part if row["source"]}),
            "question_length": length_summary(raw_part, "question_length"),
            "answer_length": length_summary(raw_part, "answer_length"),
        }
    protocol = summarize(kept)
    for part in protocol.values():
        part.pop("source_hosts")
        part.pop("presidents")
    audit = {
        "raw_splits": dict(raw_summary, test=data["test"]),
        "mapped_source_groups_raw_overlap": overlap([row for row in raw_train + test if row["source"]], "official_split", "group"),
        "eligible_train_rows": len(eligible), "train_pool": len(train_pool), "validation_pool": len(val_pool),
        "protocol": protocol,
        "article_group_cross_partition": overlap(kept, "split", "group"),
        "near_or_exact_links_removed": dict(sorted(Counter(edge[2] for edge in edges).items())),
        "removed_by_partition": dict(sorted(Counter(rows[pos]["split"] for pos in excluded).items())),
        "cross_partition_links_remaining": 0, "split_seed": SEED, "selected_fold_zero_indexed": fold,
        "sample_seeds": {"train": SEED + 1, "validation": SEED + 2},
    }
    write_json(out_dir / "qnli_stats.json", audit)
    print(f"QNLI: {Counter(row['split'] for row in kept)}", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="contains raw/, processed/ and figures/ (default: project data/)")
    parser.add_argument("--part", choices=["all", "source", "target"], default="all")
    parser.add_argument("--offline", action="store_true", help="verify the local originals without a network fallback")
    args = parser.parse_args()
    config = read_config()
    raw_dir = args.data_dir / "raw"
    out_dir = args.data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.part in ("all", "source"):
        records = ensure_inputs(raw_dir, config["qevasion"]["files"], not args.offline)
        prepare_source(raw_dir, out_dir)
        outputs = [path for path in sorted(out_dir.glob("qevasion_*")) if path.name != "qevasion_manifest.json"]
        write_json(out_dir / "qevasion_manifest.json", {"raw_files": records, "generated_files": {path.name: sha256_file(path) for path in outputs}})
    if args.part in ("all", "target"):
        records = ensure_inputs(raw_dir, config["qnli"]["files"], not args.offline)
        prepare_target(raw_dir, out_dir)
        outputs = [path for path in sorted(out_dir.glob("qnli*")) if path.name != "qnli_manifest.json"]
        write_json(out_dir / "qnli_manifest.json", {"raw_files": records, "generated_files": {path.name: sha256_file(path) for path in outputs}})


if __name__ == "__main__":
    main()
