from capstudy.vocab import END, PAD, START, UNK, Vocabulary, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("A row of United States Army jets.") == [
        "a", "row", "of", "united", "states", "army", "jets"]


def test_tokenize_keeps_internal_apostrophes():
    assert tokenize("a child's toy") == ["a", "child's", "toy"]


def test_specials_come_first_and_in_order(vocab):
    assert vocab.itos[:4] == [PAD, START, END, UNK]
    assert vocab.pad_id == 0 and vocab.start_id == 1 and vocab.end_id == 2


def test_rare_words_are_dropped():
    built = Vocabulary.build(["common common common", "rare"], min_count=2)
    assert "common" in built.stoi
    assert "rare" not in built.stoi


def test_encode_wraps_with_start_and_end(vocab):
    ids = vocab.encode("a cat sitting on a wooden table", max_len=20)
    assert ids[0] == vocab.start_id
    assert ids[-1] == vocab.end_id


def test_encode_respects_max_len(vocab):
    ids = vocab.encode("a man riding a wave on a surfboard", max_len=5)
    assert len(ids) == 5
    assert ids[-1] == vocab.end_id


def test_unknown_words_map_to_unk(vocab):
    ids = vocab.encode("a zebra", max_len=10)
    assert vocab.stoi[UNK] in ids


def test_decode_stops_at_end_and_drops_specials(vocab):
    ids = [vocab.start_id, vocab.stoi["a"], vocab.stoi["cat"], vocab.end_id,
           vocab.stoi["dogs"]]
    assert vocab.decode(ids) == "a cat"


def test_encode_decode_round_trip(vocab):
    sentence = "a cat sitting on a wooden table"
    assert vocab.decode(vocab.encode(sentence, max_len=20)) == sentence


def test_save_and_load_round_trip(vocab, tmp_path):
    path = tmp_path / "vocab.json"
    vocab.save(path)
    assert Vocabulary.load(path).itos == vocab.itos
