"""Seed corpus used only when the user has no dataset of their own yet.

This module deliberately keeps its data *out* of the user's ``data.csv``.  A
user's CSV should only ever contain labels they gave themselves; the seed rows
live here and are used as a fallback so the classifier has something to learn
from on a cold start.

The split is deliberately simple and legible:

* :data:`SEED_POSITIVES` - ordinary, pronounceable, human-sounding names.
* :data:`SEED_NEGATIVES` - strings a person would likely reject: consonant
  clusters, keyboard mashes, and unpronounceable letter soup.
"""

from __future__ import annotations

from typing import List, Tuple

SEED_POSITIVES: List[str] = [
    "alice", "brian", "carol", "david", "elena", "felix", "grace", "henry",
    "iris", "jonas", "karen", "leo", "maria", "nadia", "oscar", "paula",
    "quinn", "rosa", "simon", "tessa", "ulises", "victor", "wendy", "xavier",
    "yvonne", "zachary", "amelia", "bruno", "clara", "diego", "esteban",
    "fatima", "gabriel", "hugo", "ines", "javier", "lucia", "mateo", "noa",
    "olga", "pablo", "raul", "sofia", "teo", "valeria", "yamil", "zara",
    "adrian", "beatriz", "camila", "diego", "estrella", "fernando", "guadalupe",
    "hector", "isabel", "juan", "laura", "marisol", "nestor", "octavio",
    "patricia", "ricardo", "sergio", "teresa",
]

SEED_NEGATIVES: List[str] = [
    "xkcdzx", "qzxjkw", "vvvvv", "wtfbbq", "asdfgh", "hjklmn", "zxcvbn",
    "qwerty", "plmokn", "aaaaa", "xxxttt", "zzzqqq", "bcdfg", "kkkjjj",
    "vbnmqw", "lkjhgf", "mnbvcx", "poiuyt", "trewq", "ghjkl", "cvbnm",
    "wertyu", "dfghjk", "sdfghj", "xcvbnm", "zxcvbnm", "qazwsx", "wsxedc",
    "edcrfv", "rfvtgb", "ybhnjm", "ikmnop", "ujhses", "tyghbn", "xcvbnm",
    "qweasd", "asdzxc", "zxcvbn", "poiuyt", "lkjhgf", "mnbvcx", "qwertz",
    "asdfghjkl", "zxcvbnmqw", "qqwwweeerr", "ttgbyhnujm", "lkjhgfdsazxcv",
    "mmnnbbvv", "hhjjkkll", "qazxsw", "plokij", "mnbvcxz", "lkjhgfds",
    "wertyuiop", "asdfzxcv", "1234", "abcd", "xyzzy", "aaa", "xyz",
    "kkkk", "pppp", "zz", "qq", "xx", "jj", "ll", "oo", "ii", "uu",
]

#: A broader vocabulary the generator uses to build plausible-sounding
#: letter combinations.  Kept here so the corpus lives in one place.
COMMON_WORDS: List[str] = [
    "bright", "calm", "dawn", "ember", "flint", "grove", "haven", "ivory",
    "jade", "kindle", "lumen", "meadow", "noble", "opal", "pine", "quartz",
    "river", "stone", "tide", "umber", "vale", "wave", "zephyr", "amber",
    "birch", "coral", "drift", "echo", "frost", "glade", "hazel", "isle",
    "juniper", "kelp", "lark", "moss", "north", "oak", "pearl", "quill",
    "reef", "sable", "thorn", "violet", "willow", "yarrow", "sage", "maple",
    "cobalt", "amber", "cinder", "dune", "elm", "fern", "glass", "holly",
    "iris", "lilac", "mint", "nutmeg", "olive", "plum", "rose", "thyme",
    "blue", "gold", "green", "silver", "white", "black", "red", "deep",
    "high", "long", "moon", "sun", "star", "sky", "wind", "sea", "land",
    "hill", "lake", "rain", "snow", "storm", "dawn", "dusk", "night", "day",
    "anna", "bella", "cora", "dana", "emma", "faye", "gina", "hazel",
    "iris", "jade", "kira", "lena", "mara", "nina", "olga", "pia", "rosa",
    "sara", "tara", "uma", "vera", "wanda", "yara", "zara",
    "alan", "bram", "colin", "drew", "evan", "finn", "gale", "hugo",
    "ian", "jack", "karl", "lars", "marc", "nils", "omar", "peter",
    "quinn", "ryan", "sven", "tomas", "ulrik", "victor", "wes", "yuri",
]


def seed_dataset() -> List[Tuple[str, int]]:
    """Return the seed corpus as ``(name, label)`` pairs."""
    rows: List[Tuple[str, int]] = [(n, 1) for n in SEED_POSITIVES]
    rows += [(n, 0) for n in SEED_NEGATIVES]
    return rows
