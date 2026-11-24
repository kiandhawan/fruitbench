"""Word lists for fruit-bench: fruits (with accepted spellings), game-instruction phrasings, and
the filler/meta vocabulary used by the scorer.

Splits (`make_splits()`, deterministic): the simulated user only says the 80 most familiar fruits
(COMMON_FRUITS), split 48 train / 12 val / 20 test; Moshi's replies are matched against every fruit
in FRUITS + EXTRA_RECOGNISED. Phrasings: a third of
each starter type held out for test, a few for val, the rest train. Run `python -m fruitbench.lists`
to (re)write data/splits.json.
"""

import json
import random
from pathlib import Path

# canonical name -> extra accepted spellings / aliases (the canonical name itself is always accepted;
# plurals are generated automatically by `fruit_aliases`). The canonical name is what the user says.
FRUITS: dict[str, list[str]] = {
    "apple": [], "banana": [], "orange": [], "grape": [], "strawberry": [], "blueberry": [],
    "raspberry": [], "blackberry": [], "cherry": [], "peach": [], "pear": [], "plum": [],
    "apricot": [], "nectarine": [], "mango": [], "pineapple": [], "watermelon": ["water melon"],
    "cantaloupe": ["cantaloup", "rockmelon"], "honeydew": ["honeydew melon", "honey dew"],
    "kiwi": ["kiwifruit", "kiwi fruit"], "papaya": [], "guava": [],
    "lemon": [], "lime": [], "grapefruit": ["grape fruit"], "tangerine": [], "clementine": [],
    "mandarin": ["mandarin orange"], "pomegranate": [], "fig": [], "date": [],
    "coconut": [], "avocado": [], "passion fruit": ["passionfruit"], "dragon fruit": ["dragonfruit", "pitaya", "pitahaya"],
    "lychee": ["litchi", "lichee", "lichi"], "persimmon": [], "cranberry": [], "gooseberry": [],
    "elderberry": [], "mulberry": [], "boysenberry": [], "huckleberry": [], "currant": ["redcurrant", "red currant", "blackcurrant", "black currant"],
    "quince": [], "kumquat": ["cumquat"], "starfruit": ["star fruit", "carambola"],
    "jackfruit": ["jack fruit"], "durian": [], "rambutan": [], "mangosteen": [], "longan": [],
    "plantain": [], "tamarind": [], "olive": [], "tomato": [], "blood orange": [],
    "cherimoya": ["custard apple"], "soursop": ["guanabana"], "feijoa": ["pineapple guava"],
    "loquat": [], "pomelo": ["pummelo", "pomello"], "yuzu": [], "ugli fruit": ["ugli", "ugly fruit"],
    "acai": ["açaí", "acai berry"], "goji berry": ["goji", "wolfberry"], "cloudberry": [], "lingonberry": [],
    "salmonberry": [], "marionberry": [], "loganberry": [], "tayberry": [], "jabuticaba": [],
    "physalis": ["cape gooseberry", "ground cherry", "golden berry", "goldenberry"], "prickly pear": ["cactus pear", "tuna fruit"],
    "breadfruit": ["bread fruit"], "sapodilla": ["chikoo", "chiku"], "jujube": ["red date", "chinese date"],
    "medlar": [], "pawpaw": ["paw paw"], "miracle fruit": ["miracle berry"], "salak": ["snake fruit"],
    "honeyberry": ["haskap"], "nashi pear": ["asian pear", "nashi"],
    "galia melon": ["galia"], "casaba melon": ["casaba"], "crenshaw melon": ["crenshaw"],
    "bitter melon": ["bitter gourd"], "horned melon": ["kiwano"], "winter melon": [],
    "white currant": [], "key lime": [], "meyer lemon": [], "bergamot": [], "citron": [],
    "satsuma": [], "calamansi": ["calamondin"], "finger lime": [], "sweet lime": ["mosambi"],
    "granny smith": ["granny smith apple"], "fuji apple": [], "honeycrisp": ["honeycrisp apple"],
    "gala apple": [], "crab apple": ["crabapple"], "bartlett pear": ["williams pear"], "bosc pear": [],
    "concord grape": [], "raisin": [], "prune": [], "sultana": [], "damson": ["damson plum"],
    "greengage": ["green gage"], "mirabelle": ["mirabelle plum"], "sloe": ["sloe berry"],
    "rhubarb": [], "pluot": [], "aprium": [], "tangelo": [], "minneola": [], "ackee": ["akee"],
    "noni": [], "marula": [], "baobab": [], "carob": [], "monk fruit": ["luo han guo"],
    "wax apple": ["rose apple", "java apple"], "langsat": ["lanzones"], "duku": [],
    "santol": [], "atemoya": [], "sugar apple": ["sweetsop"], "black sapote": ["chocolate pudding fruit"],
    "mamey": ["mamey sapote"], "canistel": ["egg fruit", "eggfruit"], "lucuma": [],
    "pequi": [], "cupuacu": ["cupuaçu"], "camu camu": [], "bilberry": [], "chokeberry": ["aronia"],
    "serviceberry": ["juneberry", "saskatoon"], "sea buckthorn": [], "rowan": ["rowanberry"],
    "barberry": [], "date plum": [], "kiwiberry": ["hardy kiwi"], "pepino": ["pepino melon"],
    "navel orange": [], "cara cara orange": ["cara cara"], "acerola": ["barbados cherry"], "jostaberry": [],
    "white sapote": [], "cempedak": [], "mamoncillo": ["spanish lime"], "pitanga": ["surinam cherry"],
    "ugni": ["chilean guava"], "bael": ["wood apple"], "pulasan": [], "nance": [],
}

# Recognised in Moshi's replies (valid fruits) but never said by the simulated user.
EXTRA_RECOGNISED: dict[str, list[str]] = {
    "melon": [], "pumpkin": [], "cucumber": [], "eggplant": ["aubergine"],
    "pepper": ["bell pepper"], "squash": [], "zucchini": [], "chestnut": [], "almond": [],
    "walnut": [], "peanut": [], "cashew": [], "pecan": [], "hazelnut": [], "pistachio": [],
    "macadamia": [], "rose hip": ["rosehip"], "hawthorn": [], "mayhaw": [], "pineberry": [],
    "dewberry": [], "wineberry": [], "thimbleberry": [], "jambolan": ["java plum"], "kaffir lime": [],
    "pink lady": ["pink lady apple"], "golden delicious": [], "red delicious": [], "mcintosh": [],
}
# Botanically fruits but rarely accepted in a "fruit" game; the scorer reports them separately
# under these names when Moshi says them (they still count as fruits for now; see LOG.md).
BOTANICAL_EDGE = {"tomato", "olive", "pumpkin", "cucumber", "eggplant", "pepper", "squash", "zucchini",
                  "chestnut", "almond", "walnut", "peanut", "cashew", "pecan", "hazelnut",
                  "pistachio", "macadamia", "rhubarb", "carob", "bitter melon", "winter melon"}

# The simulated user never says these (debatable as "fruit"), but they stay recognised.
USER_EXCLUDE = BOTANICAL_EDGE | {"raisin", "prune", "sultana"}

# The fruits the simulated user says: the ~80 most familiar, easy-to-recognise ones, roughly in
# order of familiarity (hand-ranked; obscure or regional fruits are still accepted from Moshi).
COMMON_FRUITS: list[str] = [
    "apple", "banana", "orange", "grape", "strawberry", "blueberry", "raspberry", "blackberry", "cherry",
    "peach", "pear", "plum", "apricot", "nectarine", "mango", "pineapple", "watermelon", "cantaloupe",
    "honeydew", "kiwi", "papaya", "guava", "lemon", "lime", "grapefruit", "tangerine", "clementine",
    "mandarin", "pomegranate", "fig", "date", "coconut", "avocado", "passion fruit", "dragon fruit",
    "lychee", "persimmon", "cranberry", "gooseberry", "elderberry", "mulberry", "boysenberry",
    "huckleberry", "currant", "quince", "kumquat", "starfruit", "jackfruit", "durian", "plantain",
    "blood orange", "pomelo", "satsuma", "key lime", "navel orange", "rambutan", "mangosteen", "tamarind",
    "acai", "goji berry", "prickly pear", "breadfruit", "lingonberry", "yuzu", "soursop", "cherimoya",
    "loquat", "longan", "jujube", "nashi pear", "crab apple", "granny smith", "concord grape",
    "meyer lemon", "cloudberry", "bilberry", "honeycrisp", "pawpaw", "damson", "greengage",
]
N_TEST_FRUITS, N_VAL_FRUITS = 20, 12

# Phrasings of the game instructions. "user" = the user will name the first fruit after Moshi
# responds; "moshi" = the user asks Moshi to start.
PHRASINGS: dict[str, list[str]] = {
    "user": [
        "Let's take turns naming fruits as fast as we can.",
        "Let's play a game: we take turns naming fruits, as fast as possible.",
        "I'll start, then you. We name fruits back and forth, quickly.",
        "Fruit game! I say a fruit, you say a different one, as fast as you can.",
        "Okay, quick game. We alternate naming fruits. I'll go first.",
        "Let's see how fast we can go: name fruits, one each, taking turns.",
        "New game: fruits, back and forth, no repeats. I'll start.",
        "We're going to take turns saying fruits, super fast. Ready? I'll go first.",
        "Let's do a speed round of fruits. You and me, alternating, no repeating.",
        "Here's a game: each of us names a fruit in turn, as quickly as we can.",
        "Name a fruit right after I do, and we keep going. I'll start.",
        "Let's play fruits. One fruit each, fast, and don't repeat any.",
        "Quick, let's trade fruit names back and forth. Me first.",
        "Fruit rally: I name one, you name one, and we keep it going.",
        "Let's go back and forth naming fruits as fast as possible. I'll begin.",
        "I want to play a game. We take turns naming fruits, really fast.",
    ],
    "moshi": [
        "Fruit game: you go first.",
        "Let's take turns naming fruits as fast as we can. You start.",
        "Let's play a game where we take turns naming fruits. You go first.",
        "Name a fruit, then I'll name one, and we keep alternating. Go!",
        "Quick game: fruits, back and forth, as fast as possible. You first.",
        "Speed round of fruits, no repeats. You start us off.",
        "We take turns saying fruits, fast. Your turn first.",
        "Let's trade fruit names as quickly as we can. You begin.",
        "Fruit game! You say one, then me, then you. Start now.",
        "Let's see how many fruits we can name, taking turns. You're up first.",
        "Back and forth with fruits, quick as you can. Kick it off.",
    ],
}

# Words that carry no game content. A fruit-less reply made only of these is "filler/meta";
# a short fruit-less reply with other words is "non-fruit" (e.g. "Carrot.").
FILLER_WORDS = set("""
okay ok o k sure yes yeah yep yup alright all right right great good nice cool awesome fine perfect
i i'm im am me my mine you you're youre your yours we we're let's lets us our it it's its is are was
be been ready go going ahead turn first start started starting next one another more again keep
play playing game games fun let do does doing done got get have has had hmm hm um uh umm uhh ah oh
eh mm mhm huh wow well so then now and or but the a an that that's thats this there here what what's
whats how how's who why when hey hi hello thanks thank please sorry name names naming say said fruit
fruits fruity think know see wait just like really very too also not no nope don't dont can can't
cant will i'll ill would could should okay okey to of for on in at with up yourself myself
gonna wanna love sounds sound interesting fast faster quick quickly as possible try trying
out idea ideas stuck hard tough tricky one pass skip give up mind blank any anything something
heard never tried tasted familiar sure guess maybe probably favorite favourite
correct incorrect wrong true exactly indeed absolutely definitely totally mmhmm mhmm uhhuh even close
""".split())

SPLITS_PATH = Path(__file__).resolve().parent.parent / "data" / "splits.json"


def plural_forms(name: str) -> list[str]:
    """Accepted plural spellings of a (possibly multi-word) fruit name: pluralise the last word."""
    *head, last = name.split(" ")
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        pl = [last[:-1] + "ies"]
    elif last.endswith(("s", "x", "ch", "sh", "o")):
        pl = [last + "es", last + "s"]
    else:
        pl = [last + "s"]
    return [" ".join(head + [p]) for p in pl]


def fruit_aliases() -> dict[str, str]:
    """Every accepted spelling (lower case, singular and plural) -> canonical fruit name."""
    out: dict[str, str] = {}
    for table in (FRUITS, EXTRA_RECOGNISED):
        for canon, aliases in table.items():
            for a in [canon, *aliases]:
                for form in (a, *plural_forms(a)):
                    out.setdefault(form.lower(), canon)
    return out


def make_splits(seed: int = 0) -> dict:
    rng = random.Random(seed)
    assert len(set(COMMON_FRUITS)) == len(COMMON_FRUITS) and all(f in FRUITS and f not in USER_EXCLUDE for f in COMMON_FRUITS)
    pool = sorted(COMMON_FRUITS)
    rng.shuffle(pool)
    nt, nv = N_TEST_FRUITS, N_VAL_FRUITS
    fruits = {"test": sorted(pool[:nt]), "val": sorted(pool[nt:nt + nv]), "train": sorted(pool[nt + nv:])}
    user_fruits = sorted(f for f in FRUITS if f not in USER_EXCLUDE)
    phr = {"train": [], "val": [], "test": []}
    for who, lst in PHRASINGS.items():
        idx = list(range(len(lst)))
        rng.shuffle(idx)
        n_test = round(len(lst) / 3)
        n_val = max(1, round(len(lst) / 8))
        for split, sl in (("test", idx[:n_test]), ("val", idx[n_test:n_test + n_val]), ("train", idx[n_test + n_val:])):
            phr[split] += [{"text": lst[i], "starter": who} for i in sorted(sl)]
    return {"seed": seed, "fruits": fruits, "phrasings": phr,
            "n_user_fruits": len(pool), "recognised_only": sorted(set(user_fruits) - set(pool))}


if __name__ == "__main__":
    s = make_splits()
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(s, indent=1))
    print({k: len(v) for k, v in s["fruits"].items()}, {k: len(v) for k, v in s["phrasings"].items()},
          "user fruits:", s["n_user_fruits"], "recognised only:", len(s["recognised_only"]))
