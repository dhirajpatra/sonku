"""Fixed spoken phrases (greetings, consent prompts). English and Hindi are complete; other
languages have the greeting and fall back to English for the rest."""
from __future__ import annotations

PHRASES: dict[str, dict[str, str]] = {
    "greet_known": {
        "en": "Hello {name}! Nice to see you again.",
        "hi": "नमस्ते {name}! आपको फिर से देखकर अच्छा लगा।",
        "bn": "নমস্কার {name}! আপনাকে আবার দেখে ভালো লাগছে।",
        "ta": "வணக்கம் {name}! உங்களை மீண்டும் பார்த்ததில் மகிழ்ச்சி.",
        "te": "నమస్కారం {name}! మిమ్మల్ని మళ్లీ చూడటం సంతోషంగా ఉంది.",
        "kn": "ನಮಸ್ಕಾರ {name}! ನಿಮ್ಮನ್ನು ಮತ್ತೆ ನೋಡಿ ಸಂತೋಷವಾಯಿತು.",
        "ml": "നമസ്കാരം {name}! നിങ്ങളെ വീണ്ടും കണ്ടതിൽ സന്തോഷം.",
        "mr": "नमस्कार {name}! तुम्हाला पुन्हा भेटून आनंद झाला.",
        "gu": "નમસ્તે {name}! તમને ફરી મળીને આનંદ થયો.",
        "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ {name}! ਤੁਹਾਨੂੰ ਦੁਬਾਰਾ ਮਿਲ ਕੇ ਖੁਸ਼ੀ ਹੋਈ।",
        "or": "ନମସ୍କାର {name}! ଆପଣଙ୍କୁ ପୁଣି ଦେଖି ଖୁସି ଲାଗିଲା।",
    },
    "listening": {"en": "Yes?", "hi": "जी?"},
    "meet_ask_name": {
        "en": "Hi, I'm Sonku! I don't think we've met. What's your name?",
        "hi": "नमस्ते, मैं सोनकू हूँ! मुझे नहीं लगता हम पहले मिले हैं। आपका नाम क्या है?",
    },
    "ask_name": {
        "en": "By the way, I don't think we've met. What's your name?",
        "hi": "वैसे, मुझे नहीं लगता हम पहले मिले हैं। आपका नाम क्या है?",
    },
    "ask_name_again": {"en": "Sorry, I didn't catch your name. Could you say it again?", "hi": "माफ़ कीजिए, मैं आपका नाम नहीं सुन पाया। क्या आप दोबारा बता सकते हैं?"},
    "skip_name": {"en": "No problem, we can skip that. What can I do for you?", "hi": "कोई बात नहीं। मैं आपकी क्या मदद कर सकता हूँ?"},
    "ask_consent": {
        "en": "Nice to meet you, {name}! May I remember your face so I can greet you next time? Please say yes or no.",
        "hi": "आपसे मिलकर खुशी हुई, {name}! क्या मैं आपका चेहरा याद रख लूँ ताकि अगली बार आपको पहचान सकूँ? कृपया हाँ या ना कहें।",
    },
    "declined": {"en": "Okay, I won't remember your face. How can I help?", "hi": "ठीक है, मैं आपका चेहरा याद नहीं रखूँगा। मैं कैसे मदद करूँ?"},
    "enrolled": {"en": "Thanks, {name}! I'll recognise you next time.", "hi": "धन्यवाद, {name}! अगली बार मैं आपको पहचान लूँगा।"},
    "already_known": {"en": "Oh, I already know you, {name}!", "hi": "अरे, मैं तो आपको पहले से जानता हूँ, {name}!"},
    "enroll_fail": {"en": "I couldn't see your face clearly. Let's try again another time.", "hi": "मैं आपका चेहरा साफ़ नहीं देख पाया। फिर कभी कोशिश करते हैं।"},
    "forgot": {"en": "Done. I've forgotten your face and our conversations.", "hi": "हो गया। मैंने आपका चेहरा और हमारी बातचीत भुला दी है।"},
    "forgot_none": {"en": "I don't have anything stored about you.", "hi": "मेरे पास आपके बारे में कुछ भी सहेजा हुआ नहीं है।"},
    "bye": {"en": "Goodbye! Say my name whenever you need me.", "hi": "अलविदा! ज़रूरत हो तो मेरा नाम पुकारिए।"},
    "llm_down": {"en": "Sorry, I can't reach my brain right now. Please try again in a moment.", "hi": "माफ़ कीजिए, अभी मैं जवाब नहीं दे पा रहा। थोड़ी देर बाद कोशिश कीजिए।"},
}


def phrase(key: str, lang: str = "en", **fmt) -> str:
    table = PHRASES[key]
    return table.get(lang, table["en"]).format(**fmt)


def phrase_lang(key: str, lang: str) -> str:
    """Language the phrase is actually written in (English when that language has no translation)."""
    return lang if lang in PHRASES[key] else "en"
