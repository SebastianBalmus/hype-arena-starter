from arena.lexicon import Lexicon, NameIndex, attribute_words, has_attribute_words

L = Lexicon.load("data/lexicon.json")


def test_canonical_and_mentions():
    assert L.canonical("hooly   ai pro") == "Hooly AI Pro" and L.canonical("HoolyAIPro") == "Hooly AI Pro"
    assert [m.subject for m in L.mentions("Hooly AI Pro costs more than Hooly.")] == ["Hooly AI Pro", "Hooly"]
    assert [m.subject for m in L.mentions("The Workspace plan and Hooly I/O 2026")] == ["Hooly Workspace", "Hooly I/O 2026"]
    assert L.mentions("Hoolyx is not a name") == []


def test_mention_details():
    text = "Hooly Inc. and Hooly's board met; HOOLY AI PRO"
    ms = L.mentions(text)
    assert [m.subject for m in ms] == ["Hooly", "Hooly Board", "Hooly AI Pro"]
    assert text[ms[0].start:ms[0].end] == ms[0].surface == "Hooly Inc."
    assert L.canonical("nonsense") is None
    assert "soc2_certified" in L.attributes("Hooly") and L.attributes("zzz") == []
    assert "Hooly I/O 2026" in L.events and L.subjects["Hooly I/O 2026"]


def test_attribute_words():
    assert attribute_words("soc2_certified") == "soc2 certified"
    assert has_attribute_words("Hooly has never been SOC2-certified.", "soc2_certified")
    assert has_attribute_words("Hooly is SOC 2 certified", "soc2_certified")
    assert has_attribute_words("Hooly is soc2certified", "soc2_certified")
    assert not has_attribute_words("Hooly passed a SOC 2 audit", "soc2_certified")
    assert has_attribute_words("Hooly had major-outage", "had_major_outage")


def test_name_index():
    ni = NameIndex.from_values(["Gavin Bellson", "Nimbus Cloud", "Berlin"], L)
    assert ni.canonical("Gavin") == "Gavin Bellson" and ni.canonical("Berlin") == "Berlin"
    assert ni.canonical("Nimbus") == "Nimbus Cloud" and ni.canonical("Zed") == "Zed"
    assert [n for _, _, n in ni.find("Gavin Bellson moved to Berlin")] == ["Gavin Bellson", "Berlin"]
    assert [n for _, _, n in ni.find("gavinbellson, BERLIN")] == ["Gavin Bellson", "Berlin"]
    assert ni.find("Berliner") == []
