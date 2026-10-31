# Common knowledge about Earth and everyday human life

Knowledge checkpoint: **2026-10-31**

This reference layer teaches concepts the game does not explain. It is not automatically Teyvat canon and is separate from Hu Tao's autobiography and from facts the user teaches her personally.

## Dataset contents

- 262 individual facts
- 154 human-body, anatomy and daily-function records
- 50 psychology, feelings and relationship records
- 12 language, slang and social-convention records
- 34 Teyvat canon reference records
- 12 reasonable assumptions about ordinary human-like life in Teyvat
- 25 source records

## Coverage

Anatomy and organs; eating and drinking; digestion, nutrient absorption, stool, faeces, poop, defecation and flatulence; kidneys, urine, pee, urination and the urinary tract; toilets, washrooms, bathrooms, restrooms and potty vocabulary; hygiene and handwashing; sleep and common bodily functions; biological sex, gender, identity, orientation and intersex variations; reproductive anatomy, puberty, menstruation, fertilisation and pregnancy; contraception, STIs, sexual health and consent; emotions, friendship, romance, affection, intimacy, dating, boundaries and grief; profanity, adult language, pornography and media literacy.

Sexual and reproductive concepts are described in a non-graphic educational manner. The dataset includes consent, bodily autonomy, privacy and the distinction between staged adult media and real relationships.

## Evidence labels

- `earth_human_baseline`: general human knowledge. Facts labelled established have source references.
- `social_knowledge`: social and psychological concepts, with qualifications.
- `teyvat_canon`: facts established by released game content or official publications.
- `teyvat_assumption`: a practical default for ordinary human-like Teyvat residents where the game has not supplied an answer. These entries use `reasonable_assumption` and are **not canon**.
- `language_and_slang`: English vocabulary and its social meaning. English profanity is a translation-layer lesson, not a claim that those exact English words are spoken in Teyvat.

Do not apply ordinary human assumptions automatically to gods, adepti, spirits, Melusines, Saurians or other non-human species. Specific released canon overrides a general assumption. Do not assume Teyvat has modern Earth plumbing, internet pornography or identical social laws.

## Import

    python -m joseph.knowledge.ingest --db local/common_knowledge.sqlite

Facts are retrieved into model context when relevant. Importing them does not silently train the neural weights.
