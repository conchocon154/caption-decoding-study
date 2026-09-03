from capstudy.data import make_splits


def test_splits_partition_every_image_exactly_once():
    ids = list(range(1000))
    splits = make_splits(ids, seed=1)
    combined = [i for split in splits.values() for i in split.image_ids]
    assert sorted(combined) == ids
    assert len(combined) == len(set(combined))


def test_no_image_appears_in_two_splits():
    splits = make_splits(list(range(1000)), seed=1)
    train, val, test = (set(splits[n].image_ids) for n in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)


def test_splits_are_deterministic_for_a_seed():
    first = make_splits(list(range(500)), seed=7)
    second = make_splits(list(range(500)), seed=7)
    assert first["test"].image_ids == second["test"].image_ids


def test_a_different_seed_gives_a_different_split():
    a = make_splits(list(range(500)), seed=1)["test"].image_ids
    b = make_splits(list(range(500)), seed=2)["test"].image_ids
    assert a != b
