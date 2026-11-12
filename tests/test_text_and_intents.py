from app.core import intents as I
from app.utils.lang import detect_lang, normalize_lang
from app.utils.text import SentenceChunker, clean_for_tts, split_sentences


def test_sentence_chunker_streaming():
    c = SentenceChunker()
    out = []
    for piece in ["Hello the", "re! It is 3", ".5 degrees. Nic", "e day"]:
        out += c.feed(piece)
    out += c.flush()
    assert out == ["Hello there!", "It is 3.5 degrees.", "Nice day"]


def test_hindi_danda_split():
    assert split_sentences("नमस्ते दोस्त। आप कैसे हैं?") == ["नमस्ते दोस्त।", "आप कैसे हैं?"]


def test_clean_for_tts():
    assert clean_for_tts("<think>hmm</think>**Hi** 😀 see https://x.y now") == "Hi see now"


def test_language_detection():
    assert detect_lang("வணக்கம்") == "ta" and detect_lang("hello") == "en"
    assert normalize_lang("ur") == "en" and normalize_lang("kn-IN") == "kn"


def test_wake_word_variants():
    w = ["sonku"]
    assert I.find_wake_word("hi sonku", w) == (True, "")
    assert I.find_wake_word("Hey Sonko what's the weather", w) == (True, "what's the weather")
    assert I.find_wake_word("son ku how are you", w) == (True, "how are you")
    assert I.find_wake_word("सोनकू नमस्ते", w)[0]
    assert I.find_wake_word("hi Shanku hello Shanku", w)[0]  # real Google STT mishearing seen on the Pi
    assert I.find_wake_word("shonku what time is it", w) == (True, "what time is it")
    assert not I.find_wake_word("what's the weather", w)[0]
    assert not I.find_wake_word("the ship sunk yesterday", w)[0]


def test_names_and_yes_no():
    assert I.extract_name("my name is asha patel") == "Asha Patel"
    assert I.extract_name("I'm Ravi") == "Ravi" and I.extract_name("Meera") == "Meera"
    assert I.extract_name("I am fine") is None and I.extract_name("what time is it today") is None
    assert I.is_yes("yes please") and not I.is_yes("no thanks") and I.is_no("I don't want that") and I.is_yes("हाँ")
    assert I.is_forget("please forget me") and I.is_goodbye("ok bye")


def test_name_stops_at_filler_and_consent_is_implied():
    said = "yeah my name is Dheeraj and can you please remember my face by taking my photos"
    assert I.extract_name(said) == "Dheeraj" and I.implies_consent(said)
    assert not I.implies_consent("you just took my photo can you please check my name is Dheeraj")
    assert I.is_yes("go ahead") and I.is_yes("yeah sure") and not I.is_yes("has it taken")


def test_generic_words_in_wake_words_do_not_trigger():
    noisy = ["sonku", "hi", "hello", "yes", "no", "please", "good morning", "okay"]
    assert not I.find_wake_word("hello there how are you", noisy)[0]
    assert not I.find_wake_word("yes please", noisy)[0]
    assert I.find_wake_word("hi Shanku", noisy)[0]
